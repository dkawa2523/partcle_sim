# Performance baselines

These scripts are explicitly invoked, non-gating measurements. Generated JSON
is machine-local evidence and should normally remain outside version control.

## P06 reference engine

`p06_baseline.py` is a small, explicitly invoked and non-gating end-to-end
measurement. It uses `load_case`, `simulate`, and `open_result`; it does not add a
second execution path or benchmark dependency.

Run the default warm baseline from the solver project:

```console
uv run --locked python -m tests.performance.p06_baseline
```

Record a larger local observation when needed:

```console
uv run --locked python -m tests.performance.p06_baseline \
  --particles 1000 --warmups 1 --repeats 5 --json p06-baseline.json
```

Measure the accepted-path allocation peak separately from timing:

```console
uv run --locked python -m tests.performance.p06_baseline \
  --particles 256 --warmups 0 --repeats 1 --trace-memory
```

The eight scenarios separate these current costs:

- uniform-field quadratic path versus nonuniform-field RK4 and its global
  enclosure stack;
- no frames versus one frame at every macro endpoint or midpoint;
- boundaryless ballistic travel versus a material-hit case with one recorded
  hit-facet ID and stick event per particle;
- boundaryless general RK4 versus the revision 3b general-RK4 material-hit
  path;
- the same material-hit path with no trajectory versus one interior frame.

Each scenario verifies its resolved path kind and records the engine, proposal,
event, and RK4-enclosure revisions, so measurements from different numerical
algorithms are not silently compared.

`coupled_rk4_engine_v7` introduced only a deterministic proposal-work
partition: chunks of 256 candidate particles, one outstanding left-first piece
per particle and wave, grouped only by exactly equal float64 target times. The
recorded P06 comparison below used `line_quadratic_rk4_first_hit_v5`, which
added the transverse certificate while proposal v3 and RK4 enclosure v1
remained unchanged. Engine v8 limits replay retention to rows overlapping
requested frames and adds the P06-U capability. Event v6 added the certified
constant-acceleration surface start-contact rule. Engine v11 / event v7
extended that rule to strict-inward Cartesian general-RK4 start contact and
active-wall residual time. Engine v12 / event v8 added the P06-RZ
signed-stage/axis-event capability; current engine v46 preserves it while
adding P06-S/P08/P09 behavior, the P10 compiled array passes, and the P11
exponential-midpoint path. It preserves P12 event semantics, but P14-P removed
that milestone's outer worker partition. All eight
baseline scenarios remain Cartesian, so this revision does not change their
proposal, enclosure, event, or trajectory numerics; revision IDs still prevent
cross-algorithm results from being silently compared.

Reported `particles_per_s` covers case loading, preparation, integration, result
publication, and completed-result opening. The enclosure and material-hit
ratios are stack-level comparisons, not isolated helper benchmarks. Result-file
bytes per particle and the manifest's solver-owned planned bytes are reported
separately; the plan is not process peak RSS.

On the same machine and 64-particle case, event v5 recorded a material median
of 0.6450654 s and boundaryless median of 0.0471402 s, a ratio of 13.6840. The
comparison uses the immediately preceding event-v4/engine-v7 material median
of 0.9666176 s and ratio of 21.77, for about 1.50x improvement. A separate v4
remeasurement was 0.9617038 s, but is not the comparison value. Relative to the
initial scalar 2.3706326 s observation, v5 is about 3.68x faster. Accepted
pieces, candidate queries, refinements, and maximum depth are now 1088, 2496,
1408, and 21. These numbers are local observations, not acceptance thresholds.
After replay retention filtering, a five-repeat measurement recorded
0.6312817 s without frames and 0.6422232 s with one interior frame; this small
timing difference is descriptive rather than an acceptance threshold.

For the revision 3b scenario the production engine aggregates accepted
particle-pieces, BVH candidate queries, dyadic refinements, and maximum
refinement depth in the result manifest. The baseline reports those values
without retaining per-particle traces. Scenarios outside the revision 3b path
report the metrics as unavailable rather than inventing zeroes.

The optional memory pass starts `tracemalloc` after `load_case` and measures
`simulate` separately from the timing pass. A calibration allocation verifies
that the active NumPy runtime exposes its data buffers to `tracemalloc`. This is
a Python/NumPy allocation peak: it excludes process RSS and native HDF5/thread
memory. Full load-to-run RSS is measured separately by the P09 script below.

Accepted replay proposals are now retained only for proposal rows whose time
interval contains a requested frame in the current macro-step. Final state,
events, and refinement counters are unchanged. On this machine, the no-frame
material case changed as follows:

| particles | before | after | reduction |
|---:|---:|---:|---:|
| 64 | 669,511 B | 356,077 B | 46.8% |
| 256 | 2,142,293 B | 913,336 B | 57.4% |
| 1024 | 7,118,215 B | 2,108,319 B | 70.4% |

After filtering, one interior frame peaked at 369,536 B, 955,316 B, and
2,286,835 B for the same sizes. These are local, non-gating traced-allocation
observations, not RSS or a one-million-particle extrapolation. This completes
the revision 3b regular-XY memory checkpoint. P06-U material P1/Q1 support and
P06-RZ force coupling, P06-S Stokes--Cunningham, and the P08 Stage 1A closure
are also complete. P09 memory/runtime layout, P10 compiled CPU, the P11
`exponential_midpoint` ability gate, the P12 event-heavy worker partition, and the P13 durable-result gate are complete.
P14 subsequently recorded the exact-only historical matrix below. Current
source distributions are realized externally into canonical internal/surface
rows; moving walls remain a separate later extension.

Absolute timings depend on the machine and filesystem and are never pytest or CI
acceptance thresholds. Generated JSON is measurement output and should normally
remain outside version control.

## P09 load, preparation, and resident-memory characterization

`p09_memory.py` materializes a boundaryless one-step ballistic case outside the
measured process. It then launches a fresh Python child for each selected scope:

- `load`: public `load_case` only;
- `prepare`: `load_case` followed by internal production preparation, used only
  to characterize the implementation and not treated as a public contract;
- `run`: public `load_case`, `simulate`, and `open_result`.

The child reads the operating system's process high-water RSS directly. This
includes Python, NumPy, HDF5, and shared-library memory and is intentionally
reported separately from the solver-owned array memory plan. Case generation is
excluded from every measured scope. No runtime dependency or background memory
monitor is added to the solver.

Run a small local observation:

```console
uv run --locked python -m tests.performance.p09_memory \
  --particles 10000 --modes fresh warm --repeats 1 \
  --json p09-memory-10k.json
```

Run the full P09 size matrix explicitly on a machine with sufficient memory and
disk space:

```console
uv run --locked python -m tests.performance.p09_memory \
  --particles 10000 100000 1000000 --modes fresh warm --repeats 1 \
  --memory-limit-mb 4096 --json p09-memory-full.json
```

`fresh` means a new Python process; the script does not claim to flush operating
system filesystem caches. `warm` performs the requested same-process warm-up
before the measured operation. This P09 script predates P10 compilation: its
mode names characterize load/prepare/run RSS and must not be relabeled as
cold- or warm-JIT measurements. Use the P10 script below for that distinction.

Every full run records load/simulate/open timing, peak and current RSS, solver
memory-plan data, case and result artifact bytes, machine/runtime identity, case
hashes, and algorithm revisions. A semantic SHA-256 covers stable manifest facts
and all final/event/series/frame/probe data while excluding timing and RSS. The
script exits nonzero if that identity differs between fresh/warm observations or
repeats for the same case.

The source-realization direct-scatter change was characterized on the same
hardware and 300k-particle case: traced peak allocation changed from
49,203,852 B to 40,810,064 B (-17.1%), and elapsed time from 0.03271 s to
0.02106 s. This is a hardware-local, non-gating observation, not an acceptance
threshold or an end-to-end solver result.

The 10k/100k/1M observations are release/manual evidence, not ordinary pytest
work. Completion and exact semantic identity are hard requirements. Absolute
seconds and RSS ratios remain descriptive until a machine baseline exists; the
report is used to explain linearity, bytes per particle, and the gap between the
owned-array plan and process RSS before P10 optimization.

## P10 compiled CPU cold/warm characterization

`p10_compiled.py` measures the single production engine in two focused cases:

- a field-heavy, event-light nonuniform regular-field RK4 run;
- a deliberately small event-heavy regular-wall RK4 run.

Every observation runs in a fresh child process with a unique, initially empty
`NUMBA_CACHE_DIR`. `cold` is that process's first production run. `warm` first
executes the same public workflow in the same process and then measures it. The
timed scope is `load_case + simulate + open_result`; individual phase times are
also recorded. The child explicitly sets `NUMBA_DISABLE_JIT=0`. Numba runs with
`fastmath=False`, `parallel=False`, and one thread for this P10
characterization.

Run the default manual pair:

```console
uv run --locked python -m tests.performance.p10_compiled \
  --field-particles 10000 --event-particles 32 \
  --repeats 3 --warmups 1 --json p10-compiled.json
```

Run a quick plumbing check when changing only the harness:

```console
uv run --locked python -m tests.performance.p10_compiled \
  --field-particles 8 --event-particles 2 --repeats 1 --warmups 1
```

The report records the semantic SHA-256 used by P09, process RSS and the
solver-owned plan separately, every manifest revision, throughput, and the
cold-over-warm ratio. It exits nonzero if cold/warm or repeated results differ
semantically, or if the current engine/tile/runtime/memory revisions are not the
expected revisions. The ratio describes JIT amortization; it is not by itself
an inter-revision engine speedup claim. Absolute time, RSS, and speedup have
no pass/fail threshold.

On one machine, with trajectories disabled and one same-process warm-up before
three measured runs, the median same-profile observations were:

| profile | work | before | P10 warm | speedup |
|---|---:|---:|---:|---:|
| regular harmonic | 2048 particles × 4 RK4 macro steps | 1.99 s | 0.03058 s | 65.1x |
| Epstein C02 | 2048 particles × 5 RK4 macro steps | 3.74 s | 0.03339 s | 112.0x |
| event-heavy regular wall | 512 particles × 4 macro steps (including material-hit intervals) | 12.8 s | 3.14395 s | 4.07x |

The event-heavy run recorded 8,704 accepted pieces, 20,480 candidate queries,
11,776 refinements, and maximum depth 21. These are local, non-gating
observations. They identify event orchestration as the P12 bottleneck after the
completed P11 integrator gate; they do not authorize a P10-only event rewrite.

A separate read-only synthetic local profile sampled 1,000 points on a P1
strip. Warm no-hint full search took 0.026/0.131/0.249/1.271 s for
100/500/1,000/5,000 cells; a correct strict-interior hint took about 0.0005 s
in each condition (51x to 2,576x). This is non-gating diagnostic evidence, not
a product benchmark. The first RK4 stage has no hint for every particle, so the
fallback was not assumed rare. P14 therefore covered realistic cell counts,
initial localization, and cross-cell motion and found a containment BVH warranted.

This focused pair is intentionally separate from P14. P14 completed the full
10k/100k/1M particle, regular/P1/Q1, realistic unstructured cell count,
initial-localization/cross-cell-motion, 0/1/5/20-hit, output-volume, and thread
matrix and any product-level performance conclusion.

## P11 exponential-midpoint fresh/warm characterization

`p11_exponential.py` expands analytic microcase C03 to the requested particle
count and selects the native `exponential_midpoint` integrator. The case uses
one `dt=0.75 s` macro step with `dt/tau=3`, so it exercises the stiff-capable
linear-relaxation path instead of merely relabeling an RK4 run. Case generation
is outside the timed child process and trajectories are disabled.

Event regression uses the method-neutral curved chord bound formed from the
velocity-enclosure width rather than the full position-box width. The fixed
material-reflection case reaches maximum refinement depth 22 and the surface
departure/return-to-source-facet case reaches 19; both have a regression cap
of 24. These depths characterize the certificate path and are not elapsed-time
thresholds. Stokes--Cunningham constant primitives and every requested C03
frame are checked against closed forms outside this timing harness.

Every observation uses the same three public APIs and a fresh child with a
unique, initially empty `NUMBA_CACHE_DIR`. `fresh` measures the first run in
that process. `warm` performs one or more untimed public runs before measuring.
The harness requires exact semantic-result identity across modes and repeats,
the resolved CPU/exponential path, the current `compiled_cpu_tile_v21`, and the
current physics-runtime, proposal, enclosure, event, and result revisions. It also
checks the current memory-plan revisions and remains a manually invoked,
non-gating measurement.

Run the default observation:

```console
uv run --locked python -m tests.performance.p11_exponential \
  --particles 10000 --repeats 3 --warmups 1 --json p11-exponential.json
```

Run a quick harness check:

```console
uv run --locked python -m tests.performance.p11_exponential \
  --particles 32 --repeats 1 --warmups 1
```

On one machine, a 1,000-particle, one-repeat observation recorded exact
fresh/warm semantic identity. Fresh simulation took 2.70640 s and peaked at
171,589,632 B process RSS; warm simulation took 0.009588 s and peaked at
171,880,448 B. The solver-owned memory plan was 5,995,672 B in both runs. The
fresh process added 93,528,064 B to its high-water mark, while the already-warm
process added 122,880 B. These values primarily expose compilation amortization
and are local descriptive evidence, not an RK4 comparison, product-scale
throughput claim, or acceptance threshold. P14 retains ownership of those
broader conclusions.

## Historical P12 event-heavy thread characterization

P12 measured the former outer-worker runtime with one event-heavy regular-wall
case at requested worker counts 1, 2, and 4. Its executable harness was removed
during the current P14-P migration when that scheduler was deleted; retaining a runnable script with obsolete
`NUMBA_NUM_THREADS=1` semantics would create a second, invalid parallel test
path. The observations below remain historical evidence and are not claims
about the current serial runtime. The parallel acceptance driver was removed
when P14-P closed negatively.

On one machine, the default run produced the following medians. The scientific payload digest matched across all nine observations;
accepted pieces, candidate queries, refinements, and maximum depth were respectively 8,704, 19,968, 11,264, and 21 for every run.

| threads | warm simulate | end-to-end | 1-thread speedup |
|---:|---:|---:|---:|
| 1 | 0.440460 s | 0.452938 s | 1.000 |
| 2 | 0.528229 s | 0.542817 s | 0.83384 |
| 4 | 0.632361 s | 0.650943 s | 0.69653 |

The immediately preceding same-machine serial simulate baseline was 3.066786 s, so that P12 one-worker implementation was 6.96x
faster. Increasing worker count was nevertheless slower in this workload. P12 therefore closed deterministic worker ownership,
bounded per-worker scratch, stable merge, compiled BVH/preclassification, and equal-time wall-prefix batching; it does not claim a
parallel speedup or product throughput threshold. Remaining Python/GIL coordination and the particle-count break-even belong to the
P14 matrix. Absolute seconds remain descriptive and are never pytest/CI gates.

P13 subsequently completed multi-segment publication, checkpoint/resume,
failure injection, and bounded writer backpressure. P14 then recorded the
former runtime's exact-only characterization described below. Neither later
milestone retroactively changes the P12 observations.

## P13 durable-result acceptance

P13 is a correctness and recovery gate, not a new elapsed-time benchmark. A 130-macro-step public-API scenario creates three
segments at the fixed 64-macro cadence and verifies the merged release, boundary, failure, series, frame, probe, and final payloads.
Post-replace failures at segment, checkpoint, and `LATEST` boundaries verify the previous/new commit meanings, automatic resume,
orphan replacement, and raw scientific identity against an uninterrupted run. The acceptance also covers an exact initial restart
before the first `LATEST`, failures at final/manifest/`_SUCCESS`/directory publication, and probabilistic-wall RNG ordinal restoration.
Recovery rejects a corrupt referenced checkpoint or latest-segment hash and inconsistent structure/cumulative counts in any segment;
same-shape value tampering in an older non-hashed segment is outside the integrity contract. It does not expose a final state before
completion. The complete verification/scenario suite contains 322 passing tests. The capacity-one queue waits for each acknowledgement,
so P13 claims bounded backpressure, not compute/I/O overlap. Power-loss durability, remote filesystems, and concurrent processes writing
the same OUT are out of scope. P14 measured this path as part of the public end-to-end workflow; it still does not isolate pure
writer bandwidth.

## P14 orthogonal release matrix

`p14_matrix.py` is the authoritative manual P14 harness. It deliberately uses a
small orthogonal matrix instead of a Cartesian product. The release preset covers
10,000, 100,000, and 1,000,000 particles; regular, P1, and Q1 fields; initial
location and cross-cell motion; 0, 1, 5, and 20 boundary hits per particle;
none, sampled, and all-particle output; fixed-topology linear time fields via a
two-snapshot static-equivalence row with scientific-payload identity; and cold
and warm compilation states.
The current harness is serial-only. The all-particle
row is limited to the short 10,000-particle case.

The synthetic release meshes contain 10,368 P1 triangles and 2,500 Q1
quadrilaterals. Those sizes are representative of the current external
reference inventory, but the generated cases are independent solver inputs:
`model_dataset/` is neither loaded nor treated as a physics definition. The
initial and cross-cell rows exercise the public solver. Same-cell accepted-hint
cost is a direct locator characterization and must not be conflated with the
separate T04 accepted-cache scenario.

Run the quick smoke preset:

```console
uv run --locked python -m tests.performance.p14_matrix \
  --suite smoke --repeats 1 --warmups 1 \
  --memory-limit-mb 1024 --json p14-smoke.json
```

Run the serial release preset:

```console
uv run --locked python -m tests.performance.p14_matrix \
  --suite release --repeats 3 --warmups 1 \
  --memory-limit-mb 8192 --json .artifacts/p14_release_v27.json
```

The child process fixes `NUMBA_NUM_THREADS=1`; the case schema has no thread-count
setting. Development and failed-run JSON stays in ignored `.artifacts/`. P14-R copies only an
accepted formal release report into the versioned `evidence/v0.1/` directory; do not version
arbitrary benchmark output.

Every observation runs `load_case`, `simulate`, and `open_result` in a fresh
process with a private initially empty Numba cache. The current harness hardening
checks completion; exact final/event/frame/probe row counts; memory-plan fit;
the required JIT environment; repeat/science identity; literal-pinned global
revisions; and nested memory/runtime revisions. Field rows require one compiled
physics-tile dispatcher signature; exact-path event rows correctly permit zero.
Conditionally applicable boundary/exponential revisions are recorded separately.
The JSON records the raw observations, medians, particle and particle-step
throughput, event work, plan and RSS bytes, artifacts, digests, revisions,
cold/warm ratios, and the 10k-to-1M regular-case timing/plan/RSS slopes.

The reported artifact-byte rate is an effective end-to-end result-artifact rate
(`artifact bytes / simulate time`), not pure writer bandwidth. Likewise, RSS
includes the Python process and native libraries while the memory plan covers
solver-owned arrays. Phase ownership can be investigated with one representative
`cProfile` run outside this matrix; it is descriptive attribution, not a reason
to add production timers or another diagnostic subsystem. Absolute seconds,
speedups, slopes, and profiles are machine-local evidence and are never CI gates.

Focused direct `PreparedFieldSet` measurements complement rather than replace
the public matrix. After specialization warm-up, a 10,000-cell P1 strip sampled
256 points in 0.000395 s initially, 0.000160 s with correct same-cell hints, and
0.000356 s after a one-cell stale-hint crossing; a 5,000-cell Q1 strip took
0.000777, 0.000330, and 0.000889 s respectively. These local measurements show
that the accepted hint remains the fastest path while the containment index
removes the interior initial/cross-cell full scan. Outside-support lookup remains
the documented O(cell-count) fallback. They are descriptive direct-kernel
evidence, not public end-to-end throughput or an acceptance threshold.

### P14 closeout observation

The release run used three observations per row on Windows 11, Python 3.12.13,
NumPy 2.5.3, Numba 0.67.0, and a host reporting 20 physical/20 logical cores.
All 23 rows completed and passed result-shape, revision, memory-fit, required-
JIT-environment, repeat, and science-identity checks. The full verification/scenario
suite subsequently passed 336 tests. The times below are machine-local
descriptions, not CI thresholds or a comparison with COMSOL.

A separate final hardened smoke passed all 7 rows. It required one compiled
physics-tile signature for field rows, allowed zero signatures for event rows
that correctly remain on the exact path, checked exact frame/probe row counts,
and pinned global plus nested memory/runtime revisions. This smoke is the
dispatcher check; it is not retroactively attributed to the 69 release
observations.

| public row | threads | simulate | particles/s |
|---|---:|---:|---:|
| regular 10k / none | 1 | 0.09565 s | 104,551 |
| regular 100k / none | 1 | 0.78387 s | 127,571 |
| regular 1M / none | 1 | 7.53404 s | 132,731 |
| regular 100k / none | 20 | 0.41705 s | 239,780 |
| regular 1M / none | 20 | 1.57073 s | 636,647 |
| P1 10,368 cells / initial | 1 | 0.6150 s | 16,259 |
| P1 10,368 cells / crossing | 1 | 0.7472 s | 13,383 |
| Q1 2,500 cells / initial | 1 | 0.7592 s | 13,171 |
| Q1 2,500 cells / crossing | 1 | 0.8949 s | 11,175 |
| event 10k / 0, 1, 5, 20 hits | 1 | 0.681 / 2.207 / 8.299 / 33.510 s | -- |
| event 10k / 20 hits | 20 | 37.728 s | -- |

The one-hit event case took 22.609 s at 100k and 226.404 s at 1M on one
worker, corresponding to about 4,417 events/s at 1M.

The regular 100k and 1M rows gained 1.880x and 4.797x at 20 workers, while
P1/Q1 10k crossing gained only 1.174x/1.147x and the 20-hit event row slowed
to 0.888x. At the P14 milestone, `threads: 1` was therefore the conservative
default. P14-P later removed the thread setting and converged production to one
single-thread compiled runtime. Regular one-worker 10k-to-1M log
slopes were 0.94818 for simulate time, 0.58736 for planned bytes, and 0.23880 for
process peak RSS. The 1M regular row peaked at 537.2 MiB with one worker and
1,143.2 MiB with 20; the 1M one-hit event row peaked at 631.1 MiB. All stayed below
the configured 8 GiB solver-owned limit, while RSS and the solver plan remain
different quantities.

At 20 workers the parallel efficiency was about 9.4% for regular 100k, 24.0%
for regular 1M, 5.9%/5.7% for P1/Q1 10k crossing, and negative for the event
row. The regular 1M solver plan rose from roughly 600 MiB to 3.68 GiB. The old
P12 `6.96x` number compared algorithmically different one-worker paths and must
not be reported as a thread speedup.

This matrix is a synthetic baseline, not a representative chamber workload.
Its event row uses an exact ballistic path, its unstructured parallel rows use
10k particles, and no row combines nonuniform forces, surface release,
material-wall curved events, many macro steps, and output. The P14-closeout outer
worker-wave is therefore a P12/P14 historical baseline, not the finished
parallel runtime. P14-P subsequently removed multithreading before P14-U.
The closeout evidence is authoritative in
[`docs/parallel_execution_plan.md`](../../docs/parallel_execution_plan.md).

Regular 10k `none/sample/all` simulations took 0.09565/0.11857/0.11737 s and
created about 2.45/2.66/4.38 MiB of result artifacts. This observed ordering must
not be read as a writer microbenchmark; the recorded byte rate is an effective
end-to-end artifact rate. Cold-to-warm ratios were 25.193x for regular, 15.869x
for P1, and 12.867x for Q1, so cold compilation must stay separated from steady
run throughput.

The release profile exposed two genuine O(particles x cells) preparation/
sampling paths and both were corrected without adding a second engine. Before
the geometry fix, one representative 1,152-cell x 1,000-table-start `cProfile`
spent 7.309 s in `_validate_table_starts`, of which 7.224 s was
`geometry._inside_any_cell`; field preparation was only 0.073 s. These are
owner-attribution observations, not permanent production timers. Field
v3 owns an indexed supported-containment lookup but deliberately retains the
exact full scan for outside/masked provisional nearest support, which therefore
remains O(cell count). Geometry v4 owns an indexed mixed-triangle/quad volume
containment query used by table-start validation, while the unchanged exact
half-space predicate remains authoritative. Locally unresolved cells are
rejected during preparation rather than hidden by an unbounded tolerance;
precomputed CPython `hypot` edge lengths preserve scalar/compiled decisions.
Memory plan v6 counts both resident indices, a 256 B/cell field-build bound,
and a 1,024 B/cell geometry-build bound. T04 remains deferred unless a later
profile identifies an accepted-state cache or remesh need; the P14 index is an
internal locator optimization, not an external preprocessor.

## P14-P serial runtime closeout

P14-P completed with multithreading rejected. After one focused correction,
regular 1M took 9.32/10.09/10.10 s at 1/2/4 threads: 4-thread speedup was
0.923x and the new one-thread path regressed 23.7% from the documented v20
one-thread value. A field-locator microkernel scaled about 3.75x, but Python/
NumPy proposal and enclosure coordination dominated end-to-end runtime.

The parallel acceptance driver and thread-only tests were deleted. P14-P closed
on engine v27 with one serial compiled runtime; case schema v2 has only
`resources.memory_limit_mb`, and `p14_matrix.py` remains the serial
performance harness. Bounded slabs, reusable workspaces, flat SoA event
wavefronts, the stackless boundary BVH, and bounded output staging remain.
Full evidence, Amdahl reasoning, and reconsideration conditions are in
[`parallel_execution_plan.md`](../../docs/parallel_execution_plan.md).

## P14-U representative-use gate

`p14u_representative.py` is the external target-use slice for the completed
serial engine. It combines an XY surface release, Epstein drag, a non-affine
electric field, gravity, a material target, and many fixed steps. The same
driver also checks a variable, axis-regular RZ field with one axis crossing.
It does not add timers or diagnostic ownership to the production solver.

Run the quick smoke preset:

```console
uv run --locked python -m tests.performance.p14u_representative \
  --suite smoke --memory-limit-mb 8192 \
  --json .artifacts/p14u_smoke_v1.json
```

Run the formal release preset:

```console
uv run --locked python -m tests.performance.p14u_representative \
  --suite release --memory-limit-mb 8192 \
  --json .artifacts/p14u_release_v1.json
```

Only the default release preset closes P14-U. It requires 10k, 100k, and 1M
particles, both `none` and sampled output, and three fresh-process observations
per row. Raw observations and medians are retained. A separate, untimed
maximum-count `none` run records cProfile owner totals and top functions; it
does not contaminate the timed observations. Custom particle counts and smoke
runs remain development evidence and set `release_gate_complete` to false.

The numerical gate refines both XY mesh axes at fixed aspect ratio and compares
regular, P1, and Q1 runs with same-layout fine references. It checks synchronized
position and velocity, hit time and position, the same target boundary, facet
interior clearance, and same-mesh facet identity. A dense trajectory sample
includes the exact source and localized hit when comparing global and path-local
field magnitudes. The RZ gate uses self-differences, a finer reference, and the
canonical input fields themselves to audit scalar/axial evenness, radial-zero
regularity, and standard-gravity radial zero. The source coverage in this gate
is the XY `line_length` interpretation; it does not validate RZ
`revolved_area` source weighting.

Every performance row requires exactly one release and one target-stick event
per particle. Sampled output must contain four requested times for the first 32
particle IDs; `none` must contain no probes. Output mode and repeat count may not
change the common scientific payload, algorithm revisions, event work, or wall
interaction counters. Peak RSS is the process high-water since worker start,
sampled after load/simulate/open; it excludes later external validation and
stays separate from the solver-owned memory plan. Absolute seconds are
descriptive, not a portable pass/fail threshold.

The formal release completed with physics catalog
`deterministic_xy_rz_catalog_v4`, boundary algorithm `point_wall_laws_v4`, and
`release_gate_complete=true`. The local report is
[`evidence/v0.1/p14u_release_v1.json`](../../evidence/v0.1/p14u_release_v1.json).
It contains 18 raw observations and six medians. The 1M-particle medians were:

| output | simulate (s) | public end-to-end (s) | result artifact (MiB) |
|---|---:|---:|---:|
| `none` | 550.43 | 553.77 | 455.1 |
| sample | 533.95 | 537.61 | 455.3 |

Across the release observations the raw peak RSS maximum was 767.3 MiB; the
1M solver-owned plan was 614.5 MiB. All rows had zero failures, the exact
requested output utility, and identical common payload digest, algorithm
revisions, event work, and wall counters across repeats and output modes within
each particle-count group. Probe
payload digests were identical across repeats within each mode; `none` and
sample intentionally have different probe payloads. XY time and regular/P1/Q1
mesh convergence, target first-hit checks, RZ time convergence, axis crossing,
input-field parity, and gravity parity passed.

The separate 1M `none` profile assigned 28.7% of owner self time to events and
26.8% to fields. No single owner dominated enough to justify another runtime or
localized production optimization, so the one compiled serial engine remains
the production path. The `none` and sample groups ran sequentially, therefore
their timing difference is not a causal estimate of sampling overhead. Seconds
are machine-local and non-gating, and this is not a COMSOL comparison. RSS was
sampled through load, simulate, and open; it excludes later external validation.
The JSON is local evidence only. P14-R owns preservation of release evidence.
The quick smoke continues to validate only driver and gate shape.

Current production is engine v46 / compiled tile v21 / proposal v10 /
event `line_quadratic_curved_capsule_periodic_first_hit_v22`, geometry v7, source v5,
physics catalog v23 / runtime v22, runtime layout v6, RK4 enclosure v2, dense path v3,
charge-stable exponential midpoint v3 / enclosure v4, result v6, canonical case/data/result schema 3,
checkpoint schema 2, field location v4, spatial gradient v2, and memory plan v16 after P15 stationary charge, P15-D shifted-Maxwellian charge, P15-E
finite-speed Epstein, P15-F collisionless Barnes ion drag, and P16
Waldmann--Gallis thermophoresis, plus the narrowly scoped B02 inertial-Brownian
path and P18-R effective-gas drag/thermophoresis sensitivities. Engine v30 localizes an unrepresentable OU row without stopping valid
neighbors. Engine v31 adds P19-L's bounded local applicability certificate.
Historical M3-C1 event v14 narrowed only eligible dense event broad-phase queries;
event v15 added the geometry-plus-roundoff budget and robust local Hermite predicates.
Event v16 additionally clears an RK4 candidate only when the integrator-owned
position Bernstein control enclosure lies strictly inside the facet half-space after
the existing budget; otherwise it retains the candidate and fails closed. It does not
change first-event or terminal semantics. Event v20 retained every
incident facet simultaneous with the first localized hit for exact, RK4,
exponential, and Brownian material paths, then applies the shared priority/combined-normal
rule; it additionally arbitrates static Cartesian-XY periodic centre crossings against
finite-radius material contact. It also proves finite-radius exact departure for every
simultaneous material candidate before omitting a zero-time residual contact, without a
position nudge or suppression of later impacts. Current event v22 / geometry v7 /
source v5 additionally apply each material group's particle-surface or particle-centre
contact geometry while retaining the physical body radius for force and boundary laws.
Memory plan v16 counts the corresponding bounded contact-mode work. These revisions do not
retroactively relabel the P11 or P14-U observations above. P14-R remote CI
remains a separate release track. The external M3-V applicability/relevance
evaluation is complete; it did not certify full trajectory equivalence. Its
three independent follow-on workstreams do not retroactively change this
performance evidence.

P16 used the same 2,000-particle P14-U smoke immediately before and after the
compiled thermophoresis branch was added. The disabled-model core payload,
event work, and solver-owned plans remained exact: 60,441,944 bytes for `none`
and 60,446,345 bytes for sample. One before observation and three after
observations showed mixed timing movement across the two output modes, so they
do not support a causal speedup or regression claim. A separate enabled
thermophoresis-only 100,000-row warm stage measured nine repeats at a
0.014549-second median (about 6.87 million rows/s); its prepared bound arrays
occupied 1,700,024 bytes. These are machine-local, non-gating diagnostics, not
a replacement for the accepted P14-U release evidence.

P18-L used the same direct warm-stage method for 100,000 fixed-charge rows and
one reused physics workspace. Across nine alternating repeats, lift disabled
had median 0.023354 s and RZ rarefied-vorticity lift enabled had median
0.0255142 s, a 1.0925 ratio. Prepared bound storage increased by 900,016 B.
This is a machine-local, non-gating stage observation, not an end-to-end release
benchmark. At P18-L closeout it did not establish COMSOL trajectory agreement.
The later M3-C1 common-P1 composite slice passes, but does not make this timing
observation a trajectory or physical-validity gate. P18-R's direct 100,000-row warm-stage observation measured
0.0267117/0.0267643 s for the old/new revision pairs (1.00197x) with identical
payload, applicability, prepared-bound bytes, and one reused workspace. This
non-gating observation does not alter the accepted P14-U baseline.

## P19-L global-first applicability observation

`p19l_local_applicability.py` uses the public case loader, simulator, and result
reader for a 4,096-particle, 20-step general-RK4 Stokes-Cunningham pair.  The
safe case is globally certifiable; changing only an unused remote field value
makes the paired case use the bounded local-cell certificate for the same
left-cell trajectories.

```console
uv run --locked python -m tests.performance.p19l_local_applicability \
  --particles 4096 --repeats 7 --json .artifacts/p19l-performance-v1.json
```

The recorded machine-local medians were 0.9276381 s global-fast and 0.9345506 s
local-fallback, a local/global ratio of 1.00745.  A separate profile observed
zero versus 20 `local_component_bounds` calls, while the scientific payloads
were bitwise identical.  The pair resolved the same 18,538,545-byte solver
plan; a separate traced-allocation observation was 9,926,561 versus 9,927,244
bytes.  Timing and traced-allocation values are descriptive, not portable
thresholds, and the latter are not RSS.  Full method, revisions, raw repeats,
memory arithmetic, and limitations are in
[`evidence/p19l/performance_v1.json`](../../evidence/p19l/performance_v1.json)
and [`evidence/p19l/README.md`](../../evidence/p19l/README.md).

## M3-C1 event v14 operational observation

The Case-A 100 nm common-P1 material candidate exposed an event-query cost
that was not a solver-time-step convergence effect. Event v13 reused the global
absolute safety enclosure for the event BVH query. By the 450-us checkpoint,
7,623,460 of its final 7,792,306 refinements had already accumulated
(97.8331703092769%). Its final query/refinement/accepted/depth counters were
16,427,517 / 7,792,306 / 8,635,211 / 16.

Event v14 uses the current dense Bernstein position/velocity bounds as the event
broad-phase query authority only for valid `rk4_dense` rows whose global field
support was independently proven. The global enclosure remains authoritative
for shortened-stage, field-support, applicability, and acceptance safety.
Invalid dense bounds or unproven global support use the global query fallback.
The v14 material candidate reported 842,927 / 11 / 842,916 / 11 and zero
failures.

The operator-observed shell wall-time on the same local machine changed from
approximately 14m13s (about 853 s) to 36.5 s, about 23.4x. These are approximate,
machine-local, non-gating observations; they are not solver-reported timings and
are not embedded in the comparison manifests.

The v14 solver-only 0.625/0.3125/0.15625-us runs reported
query/refinement/accepted counts of 206,927/0/206,927,
413,567/0/413,567, and 826,847/0/826,847. Position, velocity, and charge RMS
observed orders were 2.029875353701904, 2.0816971911764033, and
2.044084026475049; fine-pair relative L2 values were
6.099791486063973e-8, 8.321356016032579e-8, and
1.3796067752988052e-8. All three were `ORDER_EVALUATED` and the registered
self-convergence decision passed. The approximately second-order observation
is empirical for this piecewise-P1, mesh-crossing case; it neither proves nor
disproves formal fourth-order RK4 behavior. The old v13 three-step result remains
valid precision-stability history, but artificial subdivision made it unsuitable
as independent temporal-convergence evidence.

The v14 change affected only the solver event broad phase. The hash-locked
common-P1 COMSOL input, reference, and source MPH were unchanged, so no COMSOL
study rerun was required. This observation is limited to the Case-A 100 nm,
Brownian-off, common canonical exact-connectivity P1 path through the first
wafer stick; it is not a portable speed claim or evidence for native-field
parity, physical validity, Brownian runs, 30 ms, or other cases and sizes.

## B03 public-API performance and memory observation

`b03_characterization.py` is the single manual B03 closeout harness. It runs
the public `load_case -> simulate -> open_result` path in fresh workers, warms
each worker once, and keeps separate untimed `tracemalloc` runs out of the
wall-time observations.

```console
uv run --locked python -m tests.performance.b03_characterization \
  --particles 2000 20000 --repeats 3 --warmups 1 \
  --memory-limit-mb 512 --end-s 0.2 --dt-s 0.02 --tree-depth 3 \
  --json evidence/b03/performance_v1.json
```

For B05 comparisons, `--tree-depth` remains the mandatory uniform numerical
path depth. `--adaptive-max-depth` is the maximum conditional refinement near
wall, RZ-axis, or indeterminate paths; omitting it makes it equal to the base
depth and therefore exercises the fixed-depth degeneration of the same
engine. Compare fixed and adaptive runs with the same build and case inputs,
for example `--tree-depth 3 --adaptive-max-depth 3` versus
`--tree-depth 3 --adaptive-max-depth 8`. The report records both depths,
accepted pieces, candidate queries, elapsed time, and the max-depth memory
plan. This comparison does not claim that base depth 3 has the same
first-passage resolution as a uniform depth-8 path.

All 24 measured runs finished with every particle active and zero failures.
The measured B03 acceptance revisions were read from every execution manifest: engine v34,
proposal v9, event v15, catalog v16, runtime v17, compiled tile v16, memory
plan v13, and serial runtime layout v6.

| Scenario | Particles | Median warm public time (s) | Maximum process peak RSS (B) | Solver planned (B) | Accepted pieces | Axis crossings / inferred restarts |
|---|---:|---:|---:|---:|---:|---:|
| B02 XY fixed drag | 2,000 | 0.4991855 | 136,478,720 | 9,585,155 | not emitted | 0 / not applicable |
| B02 XY fixed drag | 20,000 | 5.7721926 | 185,925,632 | 63,996,905 | not emitted | 0 / not applicable |
| B03 RZ fixed drag | 2,000 | 1.2383213 | 144,617,472 | 12,375,155 | 160,000 | 0 / 0 |
| B03 RZ fixed drag | 20,000 | 11.7141711 | 207,532,032 | 91,896,905 | 1,600,000 | 0 / 0 |
| B03 effective drag + continuous charge + gravity | 2,000 | 1.2753836 | 145,387,520 | 12,375,470 | 160,000 | 0 / 0 |
| B03 effective drag + continuous charge + gravity | 20,000 | 11.8426690 | 211,341,312 | 91,897,220 | 1,600,000 | 0 / 0 |
| B03 RZ fixed drag, axis restart | 2,000 | 1.7224805 | 147,070,976 | 12,375,155 | 182,677 | 2,000 / 2,000 |
| B03 RZ fixed drag, axis restart | 20,000 | 16.3002100 | 226,316,288 | 91,896,905 | 1,826,835 | 20,000 / 20,000 |

Every row used one slab equal to its particle count, 2,048 B/row general stage
scratch, and 224 B/row stochastic-tree work. The conservative named B03 array
inventory is 646 B/row raw and 648 B/row after eight-byte rounding, leaving
1,400 B/row inside the v13 allowance. This includes the midpoint predictor and
coefficient tables, guard masks, valid-row subsets, dense-charge invariant
certificate, and axis-restart copies; the RNG root ordinal is scalar. The
manifest/component checks and measured RSS provide no evidence for a v14
memory-plan bump.

The constant-coefficient fixed pair shared physical coefficients, seed, grid,
tree, and output schedule. Across both sizes and all repeats, charge was
bitwise identical, while the 1.5 m radial-shift-corrected position and velocity
were not: their maximum differences were `9.922618282587337e-15 m` and
`1.1102230246251565e-16 m/s`. These are recorded float64 coordinate/path
roundoff observations, not relabeled as bitwise identity or used to define a
tolerance.

Full repeats, manifest revisions, planner components, candidate-query and
refinement counters, paired differences, array arithmetic, and limitations are
in [`evidence/b03/performance_v1.json`](../../evidence/b03/performance_v1.json)
and [`evidence/b03/README.md`](../../evidence/b03/README.md). The values are
machine-local and non-gating. Process RSS includes the unreset worker
high-water, while solver-planned bytes cover solver-owned arrays; neither is a
substitute for the other. This is not a COMSOL comparison, a physical-validity
claim, a portable speed threshold, isotropic 3-D Brownian evidence, or a
general state-dependent SDE order result.

### B05 conditional-refinement observation

The then-current engine v41 / event v18 / memory-plan v15 tree was remeasured on
2026-10-08 with 500 and 5,000 particles, two macro steps, two warm repeats, and
one warm-up per repeat. The measured scope remained the public
`load_case -> simulate -> open_result` path. For 5,000 particles the median
public times were:

| Scenario | Fixed depth 2 (s) | Adaptive 2 to 5 (s) | Fixed depth 5 (s) | Fixed-5 / adaptive |
|---|---:|---:|---:|---:|
| XY fixed drag | 0.167 | 0.164 | 1.005 | 6.11x |
| RZ fixed drag | 0.258 | 0.349 | 1.695 | 4.85x |
| RZ continuous charge | 0.271 | 0.358 | 1.711 | 4.78x |
| RZ axis restart | 1.017 | 1.205 | 2.862 | 2.38x |

The three clear-path adaptive runs were bitwise identical to fixed depth 2.
The axis row intentionally refined locally: it processed 89,077 accepted pieces
instead of the fixed-depth-5 row's 439,077 pieces, while both recorded 5,000
axis crossings, zero failures, and all particles active. Adaptive and uniform
depth 5 both conservatively planned 30,848,780 bytes in that row; the observed
work reduction does not weaken the worst-case memory bound.

This is evidence that conditional refinement avoids uniform depth-5 work when
only a subset of paths needs that maximum resolution. It is not evidence that
adaptive execution is always faster than the shallower depth-2 discretization:
the RZ rows paid about 1.18x to 1.35x for classification when compared with
fixed depth 2. Uniform depth 5 is a cost comparator, not a physical golden
reference. These machine-local results exclude trajectory-frame I/O and do not
define a portable performance threshold.

A final P14 smoke run covered eight rows across regular, P1, and Q1 layouts,
static and fixed-topology-linear fields, no-output and all-frame output, and a
20-hit-per-particle event row. All rows completed with the pinned revisions,
exact science-key identity, and no failures.

## M3-C2 Case-P 287-particle owner discovery

`m3c2_casep.py` is the minimal manual owner-profile harness for the accepted
common-P1 Case-P 100 nm workload. It verifies the hash-linked selection,
registration, final PASS receipt, seed allocation, candidate campaign manifest,
and performance policy before launching a worker. The first, lower-middle, and
last accepted candidate seeds are read from that allocation and must resolve to
`319032`, `319047`, and `319063`. The locked setting is 287 particles, 30 ms,
121 frames, `dt=20 us`, Brownian tree depth 3, and `geometry_rtol=1e-8`.

Exercise the full worker and report plumbing quickly with any small canonical
case, for example a temporary C01 case materialized by the verification helper:

```console
uv run --locked python -c "from pathlib import Path; from tests.verification.microcases import materialize_microcase; materialize_microcase('C01', Path('.artifacts/m3c2-casep-smoke-case'))"
uv run --locked python -m tests.performance.m3c2_casep \
  --suite smoke \
  --fixture-case .artifacts/m3c2-casep-smoke-case/case.yaml \
  --json .artifacts/m3c2-casep-smoke.json
```

The smoke report is fixture plumbing only. It never supplies Case-P performance
evidence and always leaves optimization unauthorized.

Run the real accepted 30 ms profile explicitly; it is not an ordinary pytest or
CI workload:

```console
uv run --locked python -m tests.performance.m3c2_casep \
  --suite profile \
  --campaign-root _out_m3c2/caseP_100nm_candidate_final_v1 \
  --json .artifacts/m3c2-casep-owner-profile-v1.json
```

Each seed gets a fresh child, its own initially private `NUMBA_CACHE_DIR`, and
`NUMBA_NUM_THREADS=1`. In that child the same case runs in this order: public-API
warm-up, unprofiled `load_case -> simulate -> open_result` measurement, then a
separate cProfile run of the same three APIs. Wall time, process time, process
RSS, solver memory plan, artifact bytes, work counts, cProfile self-time owner
shares, top functions, and profile/unprofiled overhead ratios are recorded.
Profiled seconds are excluded from the baseline timing.
All `chamber_particles` function rows are also retained in neutral source/line/
function form so an owner-mapping audit does not require repeating the expensive
run. NumPy, HDF5, Numba, and other native time that cannot be attributed from
cProfile is kept outside bounded owners; this profile identifies a hotspot
hypothesis and cannot by itself satisfy the later 10k wall-time gate.

For baseline identity, the driver first checks the raw candidate manifest
against the hash accepted by `final_result_receipt.json`, and checks each case
and `result/run.json` against that manifest. The worker then opens the stored
accepted result and recomputes the P14 full scientific digest in this exact
order: `final`, `release_events`, `boundary_events`, `failure_events`,
`lifecycle_series`, `frame/0..120`, then `probe/*`. The pinned digests are:

| seed | accepted full scientific digest |
|---:|---|
| 319032 | `sha256:9c0679b19a4362db7f86160376a0b6ccdb4957d5aaeeea57ee786ab3e0ad5fcb` |
| 319047 | `sha256:ba6caf7b711f76a8129d39b8c4fe3a5b4016389d468b44067166797410a86a76` |
| 319063 | `sha256:e665fc77cb6a9882d56a0733ff4fc3497ce9708ec6544775e0b9ae55f48e7eb5` |

Warm-up, unprofiled, and profiled reruns must match that seed's accepted digest,
work counts, case/data identity, and algorithm revisions exactly. The report
also records particle macro roots, OU leaf/accepted pieces, candidate queries,
refinements/depth, wall events, axis crossings, residual splits, failures, and
frame counts.

The top-level JSON contains `authority`, `conditions`, `machine`,
`observations`, `decision`, and `claim_limits`. Each observation contains the
three runs plus `accepted_baseline`, `profile_overhead`, `environment`, and
`identity`. `decision.owner_discovery_consistent` becomes true only when all
three registered seeds preserve the accepted baseline and work, name the same
predefined bounded semantic owner, and give that owner at least 25% of profiled
public end-to-end self time. Coarse `engine_orchestration`, `compiled_tile`,
result reads, or `runtime_or_dependency` buckets cannot pass that discovery
gate.

This 287-particle run is owner discovery only. The accepted
`performance.json` policy requires a stage to explain at least 25% of an
accepted-accuracy workload with at least 10,000 particles before a bounded
optimization is a candidate. Therefore `decision.optimization_authorized` is
always `false` here.

The formal v1 discovery is complete. Every warm-up, unprofiled, and profiled
rerun matched the accepted scientific payload, work, case identity, and
algorithm revisions. The unprofiled public end-to-end wall median was
45.4668 s. `integrators` dominated all three seeds with self-time shares
42.58%, 42.86%, and 42.67%, led by `curved_chord_deviation_bounds`,
`_upper_scalar_product`, and `_lerp_controls`. Because `integrators` was not a
preregistered bounded owner, the automatic decision is
`owner_discovery_consistent=false`, `10k_confirmation_required=false`, and
`optimization_authorized=false`; the result is not retroactively relabelled to
pass the gate. The durable report is
[`evidence/m3c2/caseP_100nm_owner_profile_v1/`](../../evidence/m3c2/caseP_100nm_owner_profile_v1/README.md).

The only retained performance hypothesis is one narrow chain: curved-event
enclosure and chord preparation around each Brownian leaf. It is not a required
next step for the closed M3-C2A benchmark. If an explicit product SLA opens a
separate performance work package, its deterministic 10,000-particle workload
must preserve the original 287-particle prefix and accepted accuracy settings,
and must use low-overhead stage timing rather than cProfile self-time for the
25% decision. This conditional review does not change the completed v1
decision or authorize a production edit.

### Bounded chord follow-up

A later explicit request authorized one bounded maintenance change without
reopening M3-C2A or claiming product-scale throughput. The Python row-by-axis
chord loop was replaced by one serial compiled batch, and the duplicate
event-side formula and scalar helper were removed. No benchmark timer or
alternate runtime was added.

On the same three accepted seeds, public end-to-end wall median changed from
45.4668 s to 39.8801 s (-12.29%) and `simulate` median changed from 45.4005 s
to 39.8164 s (-12.30%). Scientific payloads, revisions, and work counts were
exact for all seeds. The retained evidence and stopping decision are in
[`evidence/m3c2/caseP_100nm_chord_optimization_v1/`](../../evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md).
This result closes the bounded follow-up; it does not authorize another owner
or replace an SLA-backed 10,000-particle performance qualification.

## Current charge and durability algorithms

The current deterministic exponential-midpoint and B03 paths freeze the charge
rate `G` and Jacobian `J <= 0` at the predicted midpoint, translate that affine
law to the proposal root, and evaluate it with an `expm1`-stable exponential.
RK4 retains its explicit `h*L_Z <= 0.5` gate. No path clips charge, performs a
charge-only subcycle, or introduces a second engine; accuracy is selected with
separate `h`, `h/2`, and `h/4` runs.

Durable commits use cumulative solver work
`W = macro_step_count + accepted_particle_pieces + candidate_queries + refinements`
and `T = max(2^20, 128*N)`. The engine decides at an accepted macro barrier
when the epoch delta reaches `T`, and always commits the final macro. Revision,
resolved threshold, components, and barrier are in both the manifest and resume
identity. Output schedules and slab widths do not affect this decision. The
synchronous single-owner writer only performs the atomic segment, inactive A/B
checkpoint, and `LATEST` persistence sequence. This section describes the
algorithm only. The manual, machine-local, non-gating closeout is recorded in
[`evidence/p20_efficiency/`](../../evidence/p20_efficiency/README.md). Its
fixed-64 emulation and current-cadence runs have byte-identical assembled
public scientific payloads; its larger stable charge step demonstrates fewer
macro steps but is explicitly not an equal-accuracy comparison or portable
timing claim.

## 2026-10-09 scoped improvement baseline

The existing P14 harness accepts `--rows` to select exact preset row IDs. The
report labels this `selected_rows`, retains per-row identities and scientific
digests, and does not present it as a complete release matrix.

```powershell
uv run --locked python -m tests.performance.p14_matrix --suite release `
  --rows regular-n10000 event-h0-n10000 event-h20-n10000 --repeats 3 `
  --json ../reviews/improvement_baseline_2026-10-09.json
```

The saved [warm baseline](../../../reviews/improvement_baseline_2026-10-09.json)
uses fresh processes, private initially empty Numba caches, one thread, and one
warm-up before each observation. Three-repeat medians on this Windows machine:

| Row | simulate s | public API total s | solver plan bytes | peak RSS bytes |
|---|---:|---:|---:|---:|
| regular, 10k | 0.41986 | 0.43346 | 48,708,029 | 400,707,584 |
| event, zero hits, 10k | 0.20418 | 0.21694 | 90,211,707 | 276,824,064 |
| event, 20 hits each, 10k | 4.31827 | 4.51880 | 90,211,707 | 340,226,048 |

Scientific digests match within each row. The high-event row produced 200,000
boundary events. The [cold regular observation](../../../reviews/improvement_cold_baseline_2026-10-09.json)
is one empty-cache run: simulate 9.65291 s, public total 9.67361 s. It is not a
distribution or a portable startup guarantee. These baseline observations used engine
v44, catalog v23, physics runtime v22, compiled tile v21, and memory plan v15.
Current production supersedes them with engine v46 / event v22 / geometry v7 /
source v5 / memory plan v16; catalog v23 / physics runtime v22 / compiled tile v21
are unchanged. The stored baseline revisions and measurements remain historical evidence.

The [evidence-only profile](../../../reviews/improvement_profile_2026-10-09.json)
and its [script](../../../reviews/improvement_profile_2026-10-09.py) reuse the
P14 public-run cases. Boundary/failure buffer allocation cost about 78 us and
97 us in the zero-hit and high-event profiled cases, respectively, each with
one macro and one slab. This does not support the proposed allocation reuse
as a useful optimization in this scope. That production change is deferred.
Profile overhead and post-timing digest work are recorded; profile timings are
not throughput estimates. Long runs with many macros/restarts remain a
different workload. These observations select work; they do not certify a
product accuracy budget, SLA, COMSOL speed ratio, or all 100k/1M workloads.

The [current measured engineering scope report](../../../reviews/product_performance_acceptance_2026-10-09.json)
records default-contact before/after scientific payload identity, solver-owned plans,
machine-local timing and RSS, representative regular 100k/1M observations, and a
current-only mixed-contact 10k observation. Its small analytic checks qualify those
preset cases only. It records no approved product accuracy budget, wall-time SLA,
speedup, or regression-ratio threshold; the 100k/1M rows each have one observation.
