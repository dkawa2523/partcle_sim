# P20 operational efficiency evidence

This directory records one manual, machine-local, non-gating run of
`tests/performance/p20_efficiency.py`.  Both workloads use the public
`load_case -> simulate -> open_result` path and have no COMSOL dependency.
Case materialization and loading, result opening, validation, artifact
accounting, and payload hashing are outside the timed scope.  Each mode had one
warm-up followed by three timed `simulate` calls in alternating order.

The run was captured at `2026-10-03T16:42:25.654686+00:00` on Windows 11,
Python 3.12.13, NumPy 2.5.3, h5py 3.16.0, and Numba 0.67.0.  The machine had 20
logical CPUs; these serial warm timings are descriptive and have no portable
pass/fail threshold.

## Locked production tuple

The harness reads every result manifest and hard-asserts the following tuple;
the JSON repeats it on every measured observation.

| owner | revision |
|---|---|
| Engine | `particle_engine_v36` |
| Compiled CPU tile | `compiled_cpu_tile_v17` |
| Step proposal | `coupled_fixed_step_proposal_v10` |
| Physics runtime | `charge_jacobian_compiled_physics_runtime_v18` |
| Durable result | `durable_segmented_result_v5` |
| Event algorithm | `line_quadratic_rk4_axis_first_hit_v16` |
| Memory plan | `solver_owned_memory_plan_v13` |
| Runtime layout | `resident_soa_serial_slab_v6` |

`exponential_midpoint_revision` is required to be null for the RK4 cadence
workload and `charge_stable_exponential_midpoint_v3` for both charge workloads.
The same values are checked in the manifest's resume identity; the memory plan
and runtime layout are also checked in the nested solver memory plan.

## Work-scaled durable cadence

The identical workload used 8 stationary boundaryless ballistic particles for
4,096 macro steps of 1 ms with no frames or probes.  Every macro contributes 8
accepted particle pieces plus one macro count, for 9 work units per macro and
36,864 cumulative work units.  The latest checkpoint in each measured result
records exactly 32,768 accepted pieces, 4,096 macro steps, and zero
query/refinement work.  The legacy comparison temporarily set only the
benchmark process's cadence floor to `64 * (N + 1) = 576` and per-particle term
to zero, so it emulates the former fixed-64 partition for this guarded case.  It
is not a production option or a second writer path; both manifests retain
`cumulative_solver_work_v1` and record their resolved threshold.

| mode | warm elapsed observations (s) | median (s) | segments | artifact files | total bytes | planned memory (B) |
|---|---:|---:|---:|---:|---:|---:|
| Current default, threshold 1,048,576 | 1.9497359, 1.9150503, 1.9337154 | 1.9337154 | 1 | 6 | 317,594 | 3,546,288 |
| Fixed-64 emulation, threshold 576 | 2.7710667, 2.7786461, 2.8049942 | 2.7786461 | 64 | 70 | 3,224,886 | 3,546,288 |

The current cadence was 1.43695x faster in this observation, reduced the
segment count by 64x, removed 64 artifact files, and reduced artifact bytes by
2,907,292 B (90.1518%).  Segment/checkpoint bytes were 283,856/15,344 B for the
current cadence and 3,175,808/30,688 B for the fixed-64 emulation.

The complete public scientific payloads were byte-identical after logical
assembly.  The digest covers final state, release/boundary/failure events,
lifecycle series, frames, and probes:

`sha256:6376db3a74723b69f9e3f6190ea0b7dfd4f7063e860121ccc98e80c97bf70384`

## Continuous-charge operational utility

The second workload used 128 stationary XY particles with the production
stationary-Maxwellian continuous-charge model and
`charge_stable_exponential_midpoint_v3` over the same 32 s physical interval.
The old-admissible label refers only to the smaller `hL <= 0.5` step; both rows
use the same current v3 method, not old production code.  Both runs retained all
128 particles as active and reported zero failures.

| step | dt (s) | maximum hL | macro steps | warm elapsed observations (s) | median (s) | artifacts | planned memory (B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old-explicit-admissible | 0.25 | 0.2608516581 | 128 | 0.3588921, 0.3618991, 0.3588442 | 0.3588921 | 6 files / 127,694 B | 15,939,326 |
| Stable larger step | 2.0 | 2.0868132652 | 16 | 0.0716001, 0.0722976, 0.0702660 | 0.0716001 | 6 files / 124,102 B | 15,939,326 |

The larger stable step reduced macro steps by 112, or 8x, and gave a 5.01245x
median `simulate` speedup on this machine.

This is an operational-utility result, not an equal-accuracy comparison.  The
two discrete payload digests and final charges differ, as expected for different
step sizes.  No accuracy improvement or acceptable production step size is
inferred from the timing.  Accuracy and stability are owned by separate tests:

- `test_exponential_midpoint_charge_is_affine_exact_stiff_and_monotone`
- `test_exponential_midpoint_couples_nonlinear_charge_and_motion_at_second_order`
- `test_continuous_charge_stability_gate_is_owned_by_explicit_rk4`
- `test_aggregate_charge_coupled_time_refinement`

Production step selection still requires convergence of the requested state,
event, fate, or ensemble observables.  The benchmark only shows that the stable
charge update can complete at `hL > 0.5` and turn a larger justified step into
fewer macro steps and lower elapsed time.

## Reproduction and authority

Run from `particle_platform_redesign/solver/`:

```console
uv run --locked python -m tests.performance.p20_efficiency \
  --json evidence/p20_efficiency/metrics.json
```

The machine-readable authority is `metrics.json`.

- Driver SHA-256: `ca843a1321c75e5616d29ffb560ae44f5f0c9fa919a3875d0bf867f3c594d216`
- JSON SHA-256: `afbda6c2656ab4c606e51f6fdce558afd8726fa9cf5e72227d22813ccb5a909b`
