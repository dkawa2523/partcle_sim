# Particle trajectory platform clean-room workspace

This workspace has been reorganized for a zero-based implementation of a
semiconductor-chamber particle trajectory solver. The former implementation is
not the foundation of the new product.

## Start here

1. Read [`particle_platform_redesign/README.md`](particle_platform_redesign/README.md).
2. Read [`particle_platform_redesign/AGENTS.md`](particle_platform_redesign/AGENTS.md)
   before editing the design or future solver.
3. Use [`particle_platform_redesign/implementation_plan.md`](particle_platform_redesign/implementation_plan.md)
   for implementation order and exit criteria.
4. Use [`particle_platform_redesign/quality_tooling_plan.md`](particle_platform_redesign/quality_tooling_plan.md)
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

The independent uv project at `particle_platform_redesign/solver/` now includes
the P00--P16 foundation, B01--B03 Brownian slices, and the P18-C/I/D/L/R optional
physics revisions. The current engine includes producer-neutral YAML/HDF5 input,
XY/RZ regular/P1/Q1 fields, continuous charge and deterministic force coupling,
RK4/exponential-midpoint motion, exact and curved material events, deterministic
wall RNG, durable segmented output, checkpoint/resume, and the P14 synthetic
performance baseline.

P14-P has closed the ineffective multithreaded runtime and converged production
execution on one deterministic compiled serial engine. See
[`particle_platform_redesign/solver/docs/parallel_execution_plan.md`](particle_platform_redesign/solver/docs/parallel_execution_plan.md).
The current semantics use engine v36, compiled tile v18, proposal v10, event v16,
runtime layout v6, memory plan v13, geometry v5, physics catalog
`inertial_langevin_rz_catalog_v17`, physics runtime v19, and wall laws
`point_wall_laws_v5`. Perfect specular reflection is parameterless, non-unit
restitution is a separate law, and standard RZ gravity cannot have a radial
component. P14-U representative-use validation is complete: its formal
10k/100k/1M release gate passed while retaining the single compiled serial
production engine. The accepted machine-local report is preserved at
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
2-D conclusion is limited to the 100 nm common-P1 two-current Case-A/Case-P anchor
and the critical-boundary microcase (`2D_CRITICAL_VV_COMPLETE`); native COMSOL
fields, other sizes and ion-drag variants, three-current COMSOL trajectories, 3-D,
and general COMSOL equivalence are not certified.
