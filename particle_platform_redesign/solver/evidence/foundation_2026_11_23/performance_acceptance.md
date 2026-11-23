# Local performance and capacity acceptance

All requested existing workloads completed with native exit 0. The current source has 87 timed observations: P14 57, P14-U 18, and process workers 12. Full row times, memory values, identities and report hashes are in [performance_acceptance.json](performance_acceptance.json).

**Timing interpretation:** these are shared-host observations, not an isolated matched timing experiment. Current P14/P14-U, the failed first baseline, A/P, wheel and verification jobs overlapped portions; the replacement baseline ran under a different load/schedule. Model, numerical settings, inputs and source are controlled, but these row times do not establish causal speedup or regression.

| Workload | Baseline | Current | Verification |
|---|---|---|---|
| P14 complete orthogonal preset | 19 rows, 1 observation each | 19 rows, 3 each: 57 | Every input identity, condition/count, core and full science hash matches |
| P14-U complete release | 10k/100k/1M × none/sample × 3: 18 | Same 18 | Every core/probe hash, executed input identity, target/probe count and event-work record matches; XY/RZ gates and 1M profiles pass |
| Same-run process workers | Current-source comparison | 10k, 320 macros, 1/4 workers × none/sample × 3: 12 | Full science within each mode; core across modes; one capture per ID, zero failures, no live children |

P14 baseline timings have one observation per row; current timings below are medians of three. P14-U and process cells have three observations each. Per-stage medians are independent summaries and need not add to the median end-to-end time.

## Source and recovery boundaries

- The baseline is clean commit `6555ce3abfb8c2eb5580115c6976d77878bacc85`, with 176 files checked before/after. The task-only observer saves original returned JSON after each worker, outside its measured interval; the original full main and identity validator still execute.
- All 87 current observations bind the 59-file production/helper/config implementation roster captured before all runs and checked after. All 59 values, including all 51 production sources, are contained identically in the archived [179-file inventory](source_map.json), whose postrun values match. The 179 inventory was captured during the first process W1 interval; it is not represented as a pre-all snapshot. The [capture precision note](source_capture_scope.json) and [59-file archive](source_runtime59.json) bind the exact timestamps and file values. The 59 roster includes every installed source and six helper implementations; three test-package markers contain only docstrings and are recorded separately, without a pre-all marker claim.
- The later [harness-only followup](harness_followup_execution.json) has a 180-file map, actual 256-particle smoke and four fault regressions. All 51 installed production sources, project/lock and accepted wheel implementation remain unchanged. The large runs are not relabelled as a full 1M rerun with this modified measurement client.
- The first baseline attempt failed during profile warmup at a 261-character Windows cache temporary path and lost its unpersisted timings. None of those timings are used. The complete replacement uses a short task TEMP/TMP and durable post-return records.
- Only the separate untimed profiles were prefilled from their own completed same-source 1M-none-r0 caches: 443 files for baseline and 445 for current, with exact copied-byte hashes recorded. Their existing 32-particle warmup and full 1M simulations still ran. All timed caches remained fresh. A current 32-particle cache-load smoke passed without changing any of its 445 cache files.

Input controls compare raw case-YAML SHA-256 and the executed canonical-data identity, not HDF5 container byte equality. Core science covers final particles, release/boundary/failure events and lifecycle series. P14 full science also covers all emitted frames/probes; P14-U has zero trajectory frames and compares every requested probe.

## P14 complete preset

Public API seconds include `load_case + simulate + open_result`. Each observation uses a fresh process/private initially empty compiler cache; warm rows have one same-process warmup, cold rows have none. Imports, input creation and external validation/hash work are outside these API timers.

| Row | Before seconds (n=1) | Current seconds (median n=3) | Current owned plan MiB | Current process peak RSS MiB (max n=3) |
|---|---:|---:|---:|---:|
| event-h0-n10000 | 0.131 | 0.134 | 91.8 | 439.4 |
| event-h1-n10000 | 0.394 | 0.480 | 91.8 | 512.6 |
| event-h1-n100000 | 2.875 | 3.770 | 296.7 | 690.3 |
| event-h1-n1000000 | 33.241 | 34.990 | 654.0 | 1031.9 |
| event-h20-n10000 | 6.246 | 6.970 | 91.8 | 519.8 |
| event-h5-n10000 | 1.450 | 1.823 | 91.8 | 519.1 |
| p1-cross | 1.135 | 1.057 | 129.5 | 796.0 |
| p1-initial | 0.774 | 0.924 | 129.5 | 797.5 |
| p1-initial-cold | 97.822 | 80.402 | 129.5 | 796.0 |
| q1-cross | 1.042 | 1.220 | 127.1 | 790.9 |
| q1-initial | 1.085 | 1.382 | 127.1 | 788.8 |
| q1-initial-cold | 73.421 | 98.859 | 127.1 | 786.6 |
| regular-n10000 | 0.372 | 0.377 | 56.9 | 520.7 |
| regular-n10000-all | 0.371 | 0.516 | 60.4 | 553.8 |
| regular-n10000-cold | 37.228 | 45.027 | 56.9 | 520.4 |
| regular-n10000-sample | 0.349 | 0.411 | 58.8 | 543.7 |
| regular-n10000-time | 0.387 | 0.450 | 56.9 | 520.8 |
| regular-n100000 | 2.702 | 3.030 | 368.6 | 674.3 |
| regular-n1000000 | 27.809 | 27.799 | 741.3 | 997.0 |

## P14-U complete release

Each API timer follows the existing 32-particle same-process warmup. The separate 1M profile is outside throughput measurements. Both versions pass temporal/reference reduction, regular/P1/Q1 mesh refinement, RZ time refinement, one axis crossing at each resolution, and the smooth-versus-cusp parity audit.

| Particles | Output | Before API seconds (median n=3) | Current API seconds (median n=3) | Current owned plan MiB | Current process peak RSS MiB (max n=3) |
|---:|---|---:|---:|---:|---:|
| 10,000 | none | 6.864 | 7.508 | 129.2 | 818.8 |
| 10,000 | sample | 10.803 | 11.558 | 131.0 | 862.8 |
| 100,000 | none | 66.560 | 79.585 | 592.2 | 1117.3 |
| 100,000 | sample | 68.129 | 75.475 | 604.3 | 1189.7 |
| 1,000,000 | none | 633.131 | 721.805 | 972.7 | 1486.0 |
| 1,000,000 | sample | 631.826 | 647.984 | 984.7 | 1554.3 |

Sample means IDs 1–32 at 0.5, 1.5, 2.5 and 3.5 seconds: four probes and 128 rows, with no requested trajectories. This does not qualify full-history 1M output or every physical model at that scale.

## Process and verification stages

One serial warmup is outside timing; each measured simulation creates/closes its own team. Worker spawn/import/cache, IPC, writer and shutdown remain inside the API interval. Raw payload SHA I/O, outcome reads, science hashing/reads and explicit `verify_contents()` are recorded separately outside that interval.

| Output | Workers | API seconds | Payload SHA I/O ms | Outcome reads ms | Science hash/reads ms | Explicit verify ms | Owned plan MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1 | 32.525 | 3.506 | 93.555 | 104.631 | 59.885 | 129.2 |
| none | 4 | 23.351 | 4.774 | 107.832 | 110.038 | 71.852 | 412.3 |
| sample | 1 | 32.038 | 3.870 | 96.566 | 100.925 | 56.181 | 131.0 |
| sample | 4 | 21.773 | 5.713 | 104.569 | 126.550 | 74.278 | 413.0 |

Raw SHA I/O reads every byte in the manifest payload roster (`context.h5`, `final.h5`, committed segments); it is not every artifact-directory file. Container hashes are per-run integrity checks, while science hashes use canonical public readers. Result4/5 opening policies differ, so opening/I/O differences are not pure kernel speed measurements.

Owned plans cover solver arrays and fit the configured 8 GiB budget. Reported RSS is a single process lifetime high-water including imports/JIT/warmup, sampled after public calls and before external validation. Neither it nor the process plans are measured whole-team RSS or an operating-system hard cap. The process experiment is 10k particles; no 100k/1M process scaling is inferred.

[Baseline source](baseline_source.json), [numerical verification](numerics_verification.json), [final CI followup](remote_ci_harness_followup.json) and [report projection](performance_projection.json) retain their separate execution scopes. Native COMSOL parity and experimental physical validation are not inferred from these workloads.
