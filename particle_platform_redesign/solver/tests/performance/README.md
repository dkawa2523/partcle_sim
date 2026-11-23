# Performance measurements

Run these development clients from the locked solver project. Every workload
uses the single public engine; none changes its policy or loads an older result
format. Timings and process-memory observations are local measurements.

## Choose a measurement

| Module | Distinct current purpose | Independent check |
|---|---|---|
| `p06_baseline` | Proposal, enclosure, material-event and output costs; optional traced allocations | Requested event/output counts; microcase analytic tests run separately |
| `p09_memory` | Load/prepare/run high-water RSS and owned-array plans | Current memory plan and public scientific payload |
| `p10_compiled` | Field-heavy and event-heavy cold/warm JIT costs | Requested event counts and unchanged science; analytic tests run separately |
| `p11_exponential` | Exponential-midpoint relaxation, fresh/warm execution | Same-input science; independent C03 analytic verification run separately |
| `p14_matrix` | Orthogonal regular/P1/Q1, location, crossing, event, output and size matrix | Exact event counts and output-independent science |
| `p14u_representative` | Surface release, nonuniform forces, material target, XY/RZ refinement | Separate finer time/mesh references and target geometry |
| `b03_characterization` | Inertial Brownian fixed/continuous drag/charge and conditional refinement | Translated XY/RZ pair and axis/charge checks; OU limits run separately |
| `p19l_local_applicability` | Global versus local applicability proof costs on identical reached fields | Safe physical cohort and equal full public science |
| `process_workers` | Same-input worker-count throughput, science and cleanup | Known target capture plus separate P14-U refinement |

The names identify workload families; they do not select historical runtimes.
`process_workers` is the current same-run parallel measurement entry. The process
architecture and interpretation are owned by
[`docs/parallel_execution_plan.md`](../../docs/parallel_execution_plan.md).

## Run current workflows

```console
uv run --locked python -m tests.performance.p14_matrix --suite smoke --repeats 1 --warmups 1 --memory-limit-mb 1024 --json .artifacts/matrix.json
uv run --locked python -m tests.performance.p14u_representative --suite smoke --memory-limit-mb 8192 --json .artifacts/representative.json
uv run --locked python -m tests.performance.b03_characterization --particles 32 --repeats 1 --warmups 1 --json .artifacts/brownian.json
uv run --locked python -m tests.performance.process_workers --output .artifacts/process-measurement --particles 10000 --macro-steps 320 --workers 1 4 --repeats 3
```

Use a **new** output directory for `process_workers`. It retains editable
`input/case.yaml`, canonical `case.h5`, each result and `measurement.json`.
Inputs and conditions are separate from the executor; identity and target checks
run after timing. Timing includes `load_case + simulate + open_result`, including
worker startup, IPC, output and shutdown. One serial warmup is untimed, and every
`simulate` creates its own workers. Reports record source/input hashes and all
samples. The no-frame/no-probe process workload does not qualify output-heavy
worker scaling.

Focused measurements:

```console
uv run --locked python -m tests.performance.p06_baseline --particles 256 --warmups 1 --repeats 3 --json .artifacts/proposals.json
uv run --locked python -m tests.performance.p09_memory --particles 10000 --modes fresh warm --repeats 1 --json .artifacts/memory.json
uv run --locked python -m tests.performance.p10_compiled --field-particles 10000 --event-particles 32 --repeats 3 --warmups 1 --json .artifacts/compiled.json
uv run --locked python -m tests.performance.p11_exponential --particles 10000 --repeats 3 --warmups 1 --json .artifacts/exponential.json
uv run --locked python -m tests.performance.p19l_local_applicability --particles 4096 --repeats 7 --json .artifacts/applicability.json
```

Run `p06_baseline --trace-memory` separately from timing. Tracemalloc observes
Python/NumPy allocations, not all native memory. P09 `fresh`/`warm` describes
process execution; use the cold/warm JIT harness to specify empty/populated
compiler caches.

## Larger explicit workloads

```console
uv run --locked python -m tests.performance.p14_matrix --suite release --repeats 3 --warmups 1 --memory-limit-mb 8192 --json .artifacts/matrix-release.json
uv run --locked python -m tests.performance.p14u_representative --suite release --memory-limit-mb 8192 --json .artifacts/representative-release.json
uv run --locked python -m tests.performance.b03_characterization --particles 5000 --repeats 3 --tree-depth 2 --adaptive-max-depth 5 --json .artifacts/brownian-adaptive.json
```

P14 release includes 10k/100k/1M sizes and selected layouts, events and output
modes. `--rows` records the selected rows; a subset is not a complete matrix.
P14-U release includes 10k/100k/1M, none/sample output, time/mesh convergence and
RZ checks. Custom counts and smoke runs leave its complete-release flag false.
A separate untimed profile is not throughput evidence.

Changing `dt` or Brownian tree depth changes numerical resolution. A faster run
under changed resolution is not an equal-accuracy implementation speedup.
Uniform maximum-depth Brownian paths are cost comparators rather than physical
golden references. Scratch capacity covers maximum reachable adaptive depth.

Owned arrays, simultaneous private process memory and working-set/RSS are
different quantities. Working-set sums can count shared pages repeatedly;
sampling can miss transient peaks. Public science comparisons include final,
release, canonical boundary/failure events, lifecycle, every emitted frame and
probe. They do not compare unsaved internal paths or HDF5 container bytes.

Current cleanup execution and limits are recorded in
[`evidence/retirement_2026_11_23/performance`](../../evidence/retirement_2026_11_23/performance/README.md).
