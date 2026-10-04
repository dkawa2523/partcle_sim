# B03 performance and memory characterization

[`performance_v1.json`](performance_v1.json) is the machine-local, non-gating
record produced by
[`tests/performance/b03_characterization.py`](../../tests/performance/b03_characterization.py).
It compares the retained B02 Cartesian frozen-start path with the B03 RZ
frozen-midpoint path through the public `load_case -> simulate -> open_result`
API. It does not use COMSOL and does not define an absolute time or RSS gate.

## Method

The driver materialized cases outside the measured process, then used one
fresh worker per observation. Each worker ran one same-process warm-up before
timing the complete public API path. There were three observations for every
scenario and particle count. A separate untimed run in the first worker for
each row recorded `tracemalloc`; it is not part of the timing result.

The two particle counts were 2,000 and 20,000. Every case used seed 1,741,
`end_s=0.2`, `dt_s=0.02`, ten indexed macro steps, Brownian tree depth three
(eight declared leaves per root), a 512 MiB solver memory limit, and no
trajectory output schedule. Engine v34's indexed macro grid retained these
original decimal clock settings without creating an extra roundoff-only
terminal root. All 24 measured runs ended with every particle active and zero
failure counts.

The OS memory value is the Windows process working-set high-water read after
the measured pipeline. That counter cannot be reset, so it includes process
startup, the warm-up, Python/native libraries, and HDF5. It must not be
equated with the solver-owned memory plan.

## Warm public-API time and memory

| Scenario | Particles | Median public time (s) | Three public times (s) | Maximum process peak RSS (B) | Solver planned (B) | Slab rows | Slab stage scratch (B) | Slab tree work (B) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| B02 XY fixed drag | 2,000 | 0.4991855 | 0.4939298, 0.4991855, 0.5101852 | 136,478,720 | 9,585,155 | 2,000 | 4,096,000 | 448,000 |
| B02 XY fixed drag | 20,000 | 5.7721926 | 5.6475601, 5.7721926, 5.7759504 | 185,925,632 | 63,996,905 | 20,000 | 40,960,000 | 4,480,000 |
| B03 RZ fixed drag | 2,000 | 1.2383213 | 1.2383213, 1.2720077, 1.2157662 | 144,617,472 | 12,375,155 | 2,000 | 4,096,000 | 448,000 |
| B03 RZ fixed drag | 20,000 | 11.7141711 | 11.7950579, 11.6726733, 11.7141711 | 207,532,032 | 91,896,905 | 20,000 | 40,960,000 | 4,480,000 |
| B03 effective drag + continuous charge + gravity | 2,000 | 1.2753836 | 1.2573797, 1.2856407, 1.2753836 | 145,387,520 | 12,375,470 | 2,000 | 4,096,000 | 448,000 |
| B03 effective drag + continuous charge + gravity | 20,000 | 11.8426690 | 11.8426690, 11.9966882, 11.8381322 | 211,341,312 | 91,897,220 | 20,000 | 40,960,000 | 4,480,000 |
| B03 RZ fixed drag, axis restart | 2,000 | 1.7224805 | 1.7470394, 1.7224805, 1.6883952 | 147,070,976 | 12,375,155 | 2,000 | 4,096,000 | 448,000 |
| B03 RZ fixed drag, axis restart | 20,000 | 16.3002100 | 16.2131938, 16.4327299, 16.3002100 | 226,316,288 | 91,896,905 | 20,000 | 40,960,000 | 4,480,000 |

These values characterize this machine and workload only. In particular, the
B03/B02 time difference is not an acceptance threshold and must not be used as
a portable speed ratio.

## Stochastic and event work

The memory plan assigned 224 B/row to the depth-first stochastic tree in every
row: `32 * (depth + 4) = 32 * 7`. The derived nominal leaf count is
`particles * 10 macro steps * 8 leaves`.

| Scenario | Particles | Nominal leaves | Accepted pieces | Candidate queries | Refinements | Maximum depth | Axis crossings / inferred restarts |
|---|---:|---:|---:|---:|---:|---:|---:|
| B02 XY fixed drag | 2,000 | 160,000 | not emitted | not emitted | not emitted | not emitted | 0 / not applicable |
| B02 XY fixed drag | 20,000 | 1,600,000 | not emitted | not emitted | not emitted | not emitted | 0 / not applicable |
| B03 RZ fixed drag | 2,000 | 160,000 | 160,000 | 160,000 | 0 | 0 | 0 / 0 |
| B03 RZ fixed drag | 20,000 | 1,600,000 | 1,600,000 | 1,600,000 | 0 | 0 | 0 / 0 |
| B03 effective drag + continuous charge + gravity | 2,000 | 160,000 | 160,000 | 160,000 | 0 | 0 | 0 / 0 |
| B03 effective drag + continuous charge + gravity | 20,000 | 1,600,000 | 1,600,000 | 1,600,000 | 0 | 0 | 0 / 0 |
| B03 RZ fixed drag, axis restart | 2,000 | 160,000 | 182,677 | 216,677 | 34,000 | 17 | 2,000 / 2,000 |
| B03 RZ fixed drag, axis restart | 20,000 | 1,600,000 | 1,826,835 | 2,166,828 | 339,993 | 17 | 20,000 / 20,000 |

B02 uses its retained boundaryless Cartesian path and therefore has no
`event_refinement` ledger; the nominal stochastic leaf count remains explicit
in the evidence. B03 v1 starts a new residual stochastic root for every
accepted RZ axis crossing. The restart counts above are consequently labeled
as an implementation inference equal to the manifest's axis-crossing count,
not as a separate manifest counter.

## Paired B02/B03 characterization

The fixed-drag pair shares mass `4e-15 kg`, linear drag rate `2 s^-1`, gas
temperature `300 K`, gas velocity `[0.1, -0.05] m/s`, initial velocity
`[0.2, -0.1] m/s`, fixed zero charge, seed, time grid, tree depth, and output
schedule. B03 uses the same uniform field values after shifting the radial
coordinate by 1.5 m. Constant coefficients make the frozen-start and
frozen-midpoint coefficient values equal.

Across both particle counts and all three repeats, particle IDs and lifecycle
were identical and charge was bitwise identical. The translated position and
velocity were not bitwise identical:

- maximum translated-position difference: `9.922618282587337e-15 m`;
- largest translated-position RMS difference: `4.750637723435437e-15 m`;
- maximum velocity difference: `1.1102230246251565e-16 m/s`;
- largest velocity RMS difference: `1.0887210773532054e-17 m/s`.

These are the observed float64 coordinate/path roundoff differences. This
evidence does not relabel them as bitwise identity or set a numerical
acceptance tolerance from them.

The supported composition row resolved effective-gas linear Epstein drag,
continuous OML charge, and gravity `[0, -0.3] m/s^2`; it completed with all
particles active and zero failures. That is a configuration-completion
observation, not physical-validity evidence.

## Stage-scratch closeout

The following conservative data-buffer inventory counts phase-separated
arrays together, double-counts the returned predictor, includes arrays shared
with B02, and includes one advanced-index transient. It therefore bounds the
named B03 path additions rather than estimating the actual live-set peak.

| Named group | Raw B/row |
|---|---:|
| Predictor return | 51 |
| Predictor construction and start relaxation | 115 |
| Root guard and classification | 40 |
| Full midpoint coefficient table | 67 |
| Effective equilibrium, safety, and selection | 69 |
| Valid coefficient subset | 67 |
| Returned root batch state and safe values | 105 |
| Cubic dense-charge invariant certificate | 52 |
| Axis-restart selected copies | 80 |
| Root-interval ordinal | 0 |
| **Raw sum** | **646** |
| **Rounded to 8 B/row** | **648** |

The root interval is a scalar recursion/cohort argument used in counter-based
RNG addressing; it is not a resident per-particle column. Generic
field/physics workspaces and the pre-existing Hermite/event workspace remain
inside the established general stage allowance, while the depth-dependent
stochastic tree is the separate 224 B/row component shown above.

Every one of the 24 manifests reported memory plan v13, 2,048 B/row general
stage scratch, and `slab_proposal_scratch = slab_rows * 2,048`. The conservative
648 B/row inventory leaves **1,400 B/row** of that allowance. Separate process
RSS and `tracemalloc` observations are broader consistency observations, not a
direct meter for these NumPy buffers. The calculation and measurements provide
no basis for a memory-plan v14 bump; B03 retains v13.

## Manifest revisions and claim limits

All current revision values below were read from every execution manifest;
the report did not substitute a hard-coded current revision:

- engine `particle_engine_v34`;
- compiled tile `compiled_cpu_tile_v16`;
- proposal `coupled_fixed_step_proposal_v9`;
- event `line_quadratic_rk4_axis_first_hit_v15`;
- physics catalog `inertial_langevin_rz_catalog_v16`;
- physics runtime `inertial_langevin_compiled_physics_runtime_v17`;
- field location `field_location_v4`;
- geometry `line_boundary_stackless_volume_cell_bvh_v5`;
- result `durable_segmented_result_v4`;
- memory plan `solver_owned_memory_plan_v13` and runtime layout
  `resident_soa_serial_slab_v6`.

The evidence is limited to this machine, seed, case matrix, two particle
counts, and ten-step window. It makes no absolute performance claim, no COMSOL
comparison, no 3-D isotropic Brownian claim, no general state-dependent SDE
order claim, and no physical-validity claim for the effective-gas or
continuous-charge model.
