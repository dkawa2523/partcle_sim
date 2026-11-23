# Performance measurements

Run these development clients from the locked solver project. Every workload
uses the single public engine; none changes its policy or loads an older result
format. Timings and process-memory observations are local measurements.

## Choose a measurement

| Module | Distinct current purpose | Independent check |
|---|---|---|
| `p06_baseline` | Proposal, enclosure, material-event and output costs; optional traced allocations | Requested event/output counts; microcase analytic tests run separately |
| `p09_memory` | Load/prepare/run high-water RSS and owned-array plans | Current memory plan and public scientific payload |
| `p10_compiled` | Field-heavy, event-heavy and constant-relaxation cold/warm JIT costs | Requested event counts and unchanged science; C03 closed-form verification and other analytic tests run separately |
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
uv run --locked python -m tests.performance.process_workers --output .artifacts/process-sampled --particles 10000 --macro-steps 320 --workers 1 4 --repeats 3 --output-mode sample
```

Use a **new** output directory for `process_workers`. It retains editable
`input/case.yaml`, canonical `case.h5`, each result and `measurement.json`.
Inputs and conditions are separate from the executor; identity and target checks
run after timing. Timing includes `load_case + simulate + open_result`, including
worker startup, IPC, output and shutdown. One serial warmup is untimed, and every
`simulate` creates its own workers. Reports record source/input hashes and all
samples. `--output-mode none` has no frames/probes; `sample` uses the existing
P14-U sample ID/time policy. Each run separately measures complete-payload
SHA-256 I/O, outcome reads, scientific hashing and explicit `verify_contents()`
outside the public end-to-end interval. The none mode does not qualify sampled
or full-history output scaling.

Focused measurements:

```console
uv run --locked python -m tests.performance.p06_baseline --particles 256 --warmups 1 --repeats 3 --json .artifacts/proposals.json
uv run --locked python -m tests.performance.p09_memory --particles 10000 --modes fresh warm --repeats 1 --json .artifacts/memory.json
uv run --locked python -m tests.performance.p10_compiled --field-particles 10000 --event-particles 32 --relaxation-particles 10000 --repeats 3 --warmups 1 --json .artifacts/compiled.json
uv run --locked python -m tests.performance.p19l_local_applicability --particles 4096 --repeats 7 --json .artifacts/applicability.json
```

Run `p06_baseline --trace-memory` separately from timing. Tracemalloc observes
Python/NumPy allocations, not all native memory. P09 `fresh`/`warm` describes
process execution; use the cold/warm JIT harness to specify empty/populated
compiler caches.

The compiled measurement selects `--scenarios field event relaxation` by default.
Use `--scenarios relaxation` for the C03 exponential-midpoint case alone. It uses
the same isolated worker, timing, memory, compiled-execution and cold/warm identity
checks as the other workloads. Its configured `dt/tau=3` is an input condition;
the independent C03 analytic check establishes its numerical meaning. Each
scenario retains its own integrator and event expectations.

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

Use a new P14-U `--json REPORT` path. Each completed worker's original JSON,
including the loaded case/data identity, is atomically retained in
`REPORT.observations/` before the next worker starts. A later failure leaves
these observations without publishing a completed report; they are not a partial
release qualification or a resume format. Console-only runs have no durable
observation archive. Timed repetitions use distinct fresh caches with short
private labels. The untimed profile reuses the completed maximum-count
`none`, repeat 0 cache, so profile preparation does not repeat cold compilation.

Changing `dt` or Brownian tree depth changes numerical resolution. A faster run
under changed resolution is not an equal-accuracy implementation speedup.
Uniform maximum-depth Brownian paths are cost comparators rather than physical
golden references. Scratch capacity covers maximum reachable adaptive depth.

Owned arrays, simultaneous private process memory and working-set/RSS are
different quantities. Working-set sums can count shared pages repeatedly;
sampling can miss transient peaks. Public science comparisons include final,
release, canonical boundary/failure events, lifecycle, every emitted frame and
probe. They do not compare unsaved internal paths or HDF5 container bytes.

B03 reports the completed public memory plan and stochastic reserve without
copying its private byte formula. The existing workspace-ownership verification
counts concrete base/adaptive tree and pending-bank allocations against that
reserve. B03 retains its separate conservative transient stage-array bound;
prepared allocations and process RSS do not replace that check.

Current foundation measurements and their execution-source limits are recorded in
[`evidence/foundation_2026_11_23`](../../evidence/foundation_2026_11_23/README.md).
The preceding cleanup execution and limits are recorded in
[`evidence/current_only_2026_11_23`](../../evidence/current_only_2026_11_23/README.md).
