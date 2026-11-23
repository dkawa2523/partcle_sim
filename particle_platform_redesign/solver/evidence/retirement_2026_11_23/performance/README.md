# 当時のbenchmark実行記録

本書のcurrentと数値は、各receiptが束縛する当時のsourceに限定する。
現在の入口は[性能client一覧](../../../tests/performance/README.md)と[追加撤去記録](../../current_only_2026_11_23/README.md)を参照する。

The current workloads use one production engine and the current canonical input
and result formats. Each report below is a new execution, not a conversion of
an earlier result. These focused executions check callable workflows and their
scientific conditions; they are not a complete 100k/1M release qualification.

## Removal and retained responsibilities

- Removed `m3c2_casep` and its dedicated test: they required obsolete accepted
  campaign hashes and result revisions. Current representative profiling stays
  in `p14u_representative`; native comparison remains external V&V.
- Removed `p20_efficiency`: its fixed-64 comparison monkeypatched a retired
  writer cadence. Current memory/I/O measurements remain in P09, and analytic
  exponential/charge stability and convergence remain in verification tests.
  Its different-step utility timings did not establish equal accuracy.
- Removed obsolete thread-setting checks, exact private helper call-count
  assertions, and the Brownian scratch-size freeze. The conservative named-array
  bound, tree-capacity accounting, physical conditions and science comparisons
  remain. Profiles are observations rather than tests of private helper names.
- Brownian runs accept one or more requested particle counts. The XY control
  uses the current midpoint model; it is not a historical frozen-start runtime.
- Repaired P19's source expansion to include `contact_radius_m`. Its former
  176-byte equality covered dense controls alone; the current full path budget
  also covers retained enclosures. It now checks that the independently counted
  controls fit, that the component equals slab capacity times the full budget,
  and that certificate/candidate capacity bounds hold.
- Corrected JSON and opaque library metadata annotations in the retained
  benchmark owners. Ruff and explicit Pyrefly over `tests/performance` pass.
  The new worker harness uses only locked runtime dependencies.

The distinct current workload purposes and executable commands are in
[`tests/performance/README.md`](../../../tests/performance/README.md).
Generated software results from retired implementations are removed; real native
data, experimental records, archived environments and independent physical
references are outside this removal.

## Executed workflows

Commands ran from the locked solver project with `uv run --locked`. Reports
include the actual conditions and raw observations. These probes overlapped
other verification, so their elapsed times are not comparative speed claims.

| Workload | Executed conditions | Result and evidence |
|---|---|---|
| P06 | 8 particles, one warmup/repeat | Requested proposal/event/output shapes passed; [report](proposals.json) |
| P09 | 8 particles, fresh/warm × load/prepare/run | Six observations, memory-plan and science checks passed; [report](memory.json) |
| P10 | Field 8 / event 2, cold/warm, one repeat | Four observations and compiled-path/science checks passed; [report](compiled.json) |
| 当時のP11 | 8 particles, fresh/warm, one repeat | 旧wrapper専用reportは撤去。[当時の観測](https://github.com/dkawa2523/partcle_sim/blob/75100e74f1243b527e065c9e6a68121221408b39/particle_platform_redesign/solver/evidence/retirement_2026_11_23/performance/exponential.json)。C03の現行計測はP10 relaxationが所有する |
| P14 | Complete smoke preset, one repeat/warmup, 1024 MiB | Eight regular/P1/Q1/time-field/event/output rows passed; [report](matrix.json) |
| P14-U | Smoke, 256-particle none/sample modes | Separate time/mesh refinement, RZ, target and output identity checks passed; [report](representative.json) |
| Brownian | 32 particles, one repeat/warmup | Four XY/RZ/fixed/continuous/axis cases passed; [report](brownian.json) |
| P19 | 8 particles, one repeat per variant | Global/local safe cohort, complete science, capacity and separate profile/trace checks passed; [report](applicability.json) |

For example, the complete smoke and focused commands were:

```console
uv run --locked python -m tests.performance.p14_matrix --suite smoke --repeats 1 --warmups 1 --memory-limit-mb 1024 --json .artifacts/matrix.json
uv run --locked python -m tests.performance.p14u_representative --suite smoke --memory-limit-mb 8192 --json .artifacts/representative.json
uv run --locked python -m tests.performance.b03_characterization --particles 32 --repeats 1 --warmups 1 --json .artifacts/brownian.json
uv run --locked python -m tests.performance.p19l_local_applicability --particles 8 --repeats 1 --json .artifacts/applicability.json
```

The independent C03 closed-form public run, OU covariance/conditional splitting,
affine stiff charge limit and nonlinear coupled-charge order tests were executed
separately: [6 passed](scientific_oracles.txt). They are retained scientific
oracles; repeated new-code output alone is not an accuracy proof.

```console
uv run --locked pytest tests/scenarios/test_force_coupled_run.py::test_exponential_midpoint_c03_matches_the_closed_form tests/verification/test_stochastic.py::test_joint_ou_covariance_and_half_conditioning_match_independent_oracles tests/verification/test_integrators.py::test_exponential_midpoint_charge_is_affine_exact_stiff_and_monotone tests/verification/test_integrators.py::test_exponential_midpoint_couples_nonlinear_charge_and_motion_at_second_order -q
```

## Current same-run worker measurement

The final [worker report](process_workers.json) ran after implementation and
other project verification finished. It binds the final source files, lock,
input-generator dependencies and actual input hashes. The current generator
created the input afresh; no historical case directory or old result reader was
needed. Full input/results remain in ignored
`.artifacts/retirement_process_verified_20261123/`; the command recreates them
from the checked-in current producer.

```console
uv run --locked python -m tests.performance.process_workers --output .artifacts/process-measurement --particles 10000 --macro-steps 320 --workers 1 4 --repeats 1
```

Conditions: XY regular32×4, 10,000 realized surface particles, 0–4 s,
dt=0.0125 s / 320 macro steps, seed0, Epstein+Coulomb+gravity, no Brownian,
no trajectory frames or probes. One serial warmup was untimed; each timed run
started its own process team. Timing includes load, simulate, open, spawn, IPC,
writer and shutdown, while generation, validation and hashing are excluded.

| Workers | Public end-to-end seconds | Physical/identity checks |
|---:|---:|---|
| 1 | 29.278202 | 10,000 target captures, 0 failures, 0 live child processes |
| 4 | 18.353255 | Same complete public science, target geometry and cleanup |

For these single observations, four workers took 37.31% less elapsed time
(1.5953×). This is a local observation, not a portable scaling or significance
claim. Final, release, boundary/failure events and lifecycle payloads matched
bitwise. There were no frames/probes, so output-heavy scaling is not established.
The independently specified target checks required one retained capture per ID
at x=1, interior y and 0<t<4 s; they are distinct from worker equality.

An earlier trial was rejected because source changed during measurement. A later
intermediate run preceded final event retirement. Neither is used in the final
table. After inspecting the exact generated layout, the main agent removed both trial input/results using bounded explicit file paths; the final verified run is retained.

## Limits

Smoke/custom counts do not qualify the complete release matrix. A finer
same-method P14-U reference is a refinement check, distinct from the independent
analytic oracles above. Equal worker outputs establish execution identity, not
universal numerical or physical accuracy. No new COMSOL solve, native random-path
matching, physical adhesion/flux certification or portable timing threshold is
claimed. Whole-process RSS is distinct from owned-array memory plans.
