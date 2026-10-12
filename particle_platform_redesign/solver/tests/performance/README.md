# Manual performance measurements

These harnesses exercise the single production engine. Run them explicitly from
the solver's locked uv project; they are not ordinary pytest timing gates.
Absolute seconds and process memory are machine-local observations.
Generated development output belongs in ignored `.artifacts/`. Accepted evidence
is preserved with its input, source, policy, raw samples, and scientific identity.

Current process architecture, source qualifications, measured speedups and
negative results have one owner:
[`docs/parallel_execution_plan.md`](../../docs/parallel_execution_plan.md).
The public execution resource is `simulate(..., workers=1)` / CLI `--workers`.
Case files contain no thread setting. Macro synchronization remains fixed at one
step; a configurable multi-step window is a future proposal.

## Available harnesses

| Module | Measurement scope |
|---|---|
| `p06_baseline` | Small public-API proposal, enclosure, material-event and output costs; optional traced allocations |
| `p09_memory` | Load/prepare/run process high-water RSS versus owned-array plans |
| `p10_compiled` | Field-heavy and small event-heavy cold/warm JIT observations |
| `p11_exponential` | Analytic exponential-midpoint relaxation, fresh/warm execution |
| `p14_matrix` | Serial regular/P1/Q1, localization, crossing, event/output and 10k/100k/1M matrix |
| `p14u_representative` | Surface release, nonuniform forces, material target, multiple steps, XY/RZ convergence |
| `b03_characterization` | Inertial Brownian fixed/continuous drag/charge, RZ axis and conditional tree refinement |
| `p19l_local_applicability` | Globally certified versus local-cell applicability path |
| `p20_efficiency` | Durable commit cadence and stable continuous-charge efficiency |
| `m3c2_casep` | Hash-bound historical accepted COMSOL Case-P owner discovery |

These serial harnesses do not implement another parallel engine. For same-run
process comparisons use the preserved R64/R65/R66 external measurement scripts
and receipts linked below; do not relabel serial harness timings as worker scaling.
Harness-specific revision expectations reject stale results rather than silently
reusing them. Check the module's `--help` before changing a preset.

## Quick measurements

```console
uv run --locked python -m tests.performance.p14_matrix --suite smoke --repeats 1 --warmups 1 --memory-limit-mb 1024 --json .artifacts/p14-smoke.json
uv run --locked python -m tests.performance.p14u_representative --suite smoke --memory-limit-mb 8192 --json .artifacts/p14u-smoke.json
uv run --locked python -m tests.performance.b03_characterization --particles 32 --repeats 1 --warmups 1 --json .artifacts/b03-smoke.json
```

Smoke runs check selected workflow/identity behavior and do not qualify a release
size matrix. P14 has `--rows` for exact preset row IDs; the report records
`selected_rows`, so selected rows must not be described as a complete matrix.

Focused observations can use smaller particle counts:

```console
uv run --locked python -m tests.performance.p06_baseline --particles 256 --warmups 1 --repeats 3 --json .artifacts/p06.json
uv run --locked python -m tests.performance.p09_memory --particles 10000 --modes fresh warm --repeats 1 --json .artifacts/p09.json
uv run --locked python -m tests.performance.p10_compiled --field-particles 10000 --event-particles 32 --repeats 3 --warmups 1 --json .artifacts/p10.json
uv run --locked python -m tests.performance.p11_exponential --particles 10000 --repeats 3 --warmups 1 --json .artifacts/p11.json
uv run --locked python -m tests.performance.p19l_local_applicability --particles 4096 --repeats 7 --json .artifacts/p19l.json
uv run --locked python -m tests.performance.p20_efficiency --json .artifacts/p20.json
```

Use `p06_baseline --trace-memory` separately from elapsed-time measurement.
Tracemalloc observes exposed Python/NumPy allocations and excludes process RSS,
native library and HDF5 memory. P09's `fresh`/`warm` mode names predate the compiled
CPU milestone and do not by themselves mean empty-/populated-JIT-cache execution.

## Larger explicit workloads

Run on a machine with adequate memory and disk space:

```console
uv run --locked python -m tests.performance.p14_matrix --suite release --repeats 3 --warmups 1 --memory-limit-mb 8192 --json .artifacts/p14-release.json
uv run --locked python -m tests.performance.p14u_representative --suite release --memory-limit-mb 8192 --json .artifacts/p14u-release.json
uv run --locked python -m tests.performance.b03_characterization --particles 5000 --repeats 3 --tree-depth 2 --adaptive-max-depth 5 --json .artifacts/b03-adaptive.json
```

P14 release covers regular/P1/Q1 meshes, initial and cross-cell location,
0/1/5/20-hit rows, no/sample/all output, and a fixed-topology time-field row.
P14-U's default release preset includes 10k/100k/1M, none/sample modes, time and
mesh convergence, target first-hit and RZ checks. Custom particle counts and smoke
runs leave its complete-release flag false. The profile run is separate and
untimed; profile seconds are not throughput measurements.

B03 distinguishes base tree depth from conditional adaptive depth. Clear adaptive
paths can reproduce the shallower discretization while boundary/axis candidates
refine locally. Uniform maximum-depth runs are cost comparators, not physical
golden references. Fewer observed refined rows do not reduce worst-case scratch
capacity. Changing `dt` or depth changes the numerical method's resolution;
time saved that way is not an equal-accuracy implementation speedup.

P20 compares work-based durable cadence and stable-charge execution. Its larger
charge step is an efficiency observation with a different step size, not an
equal-accuracy speedup. Complete science and durability semantics remain separate
from elapsed-time claims.

## Hash-bound historical Case-P owner discovery

`m3c2_casep` is tied to the historical accepted campaign, allocation and revisions.
Its smoke fixture tests report plumbing only:

```console
uv run --locked python -m tests.performance.m3c2_casep --suite smoke --fixture-case PATH/TO/case.yaml --json .artifacts/m3c2-smoke.json
```

The historical profile entry is:

```console
uv run --locked python -m tests.performance.m3c2_casep --suite profile --campaign-root _out_m3c2/caseP_100nm_candidate_final_v1 --json .artifacts/m3c2-owner.json
```

It checks authority hashes and exact result revisions before executing. A stale
campaign/revision rejection must not be bypassed to claim a current comparison.
The 287-particle owner discovery did not authorize a product-scale optimization;
its later bounded chord follow-up closed separately. Original decisions remain in
[owner-profile evidence](../../evidence/m3c2/caseP_100nm_owner_profile_v1/README.md)
and [chord evidence](../../evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md).
Current Case A/P Brownian comparisons use R65 below.

## Current retained evidence

- [R64: v62 process dispatch, scientific identity, use matrix, Windows/Linux and installed-wheel qualification](../../evidence/process_dispatch_memory_2026_10_12/README.md)
- [R65: full-step Case A/P Brownian, 1/2/4-worker performance and scoped saved-COMSOL population comparison](../../evidence/comsol_brownian_fullstep_parallel_2026_10_12/README.md)
- [R66: P14-U 10,000 particles, 320 steps, 1/4-worker observation](../../evidence/p14_10000_320step_process_2026_10_13/README.md)

Read each receipt's numerical settings, actual output, timing scope and source
binding. R65's three warm seeds per case differ from R66's three repeats of one
seed. They are descriptive medians, not portable significance or SLA guarantees.
Check complete public science: final, release/boundary/failure events, lifecycle,
and every emitted frame/probe. Unsaved internal trajectories are not directly
verified by this comparison. Stored event-batch order is distinct from the public
canonical order, and HDF5 byte identity is not the science contract.

Owned-array budget, sampled simultaneous private commit and working-set/RSS are
different quantities. Sampling can miss transient peaks; working-set sums can
repeat shared pages. A new process with a private empty Numba cache does not flush
OS filesystem caches. Every `simulate` launches its own workers; cached parent
repeats do not imply a worker pool shared across calls. Include startup, callback
compile, IPC, writing and shutdown in end-to-end worker claims.

Historical elapsed times, deleted thread-runtime trials and intermediate progress
are retained in evidence and Git history rather than duplicated as current
qualification here. No extra engine, production timer, autotuner or diagnostic
framework is introduced by these harnesses.
