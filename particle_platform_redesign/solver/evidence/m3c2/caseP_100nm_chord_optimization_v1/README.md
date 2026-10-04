# Case-P curved-chord optimization

Status: `PASS_RETAINED`.

This is the finite follow-up to the accepted Case-P owner discovery. It does
not reopen the completed COMSOL comparison and does not add a profiling or
diagnostic subsystem to production.

The only production change replaces the Python row-by-axis implementation of
`curved_chord_deviation_bounds` with one serial compiled batch. The same
component formula is now owned by `integrators.py` and reused by `events.py`;
the former scalar helper and duplicate event-side formula were removed. The
force models, time step, Brownian tree, event ordering, output schedule, public
API, and single-engine architecture are unchanged.

## Result

The accepted seeds `319032`, `319047`, and `319063` were rerun through the
three public APIs with one Numba thread. Every seed retained its accepted
scientific payload, algorithm revisions, and work counts: 1,500 macro steps,
3,444,000 OU leaves, 3,444,000 accepted pieces, and 3,444,000 candidate
queries, with no failure, wall event, axis crossing, or refinement.

| quantity | before median | after median | change |
|---|---:|---:|---:|
| public end-to-end wall | 45.4668 s | 39.8801 s | -12.29% |
| `simulate` wall | 45.4005 s | 39.8164 s | -12.30% |
| public end-to-end process | 45.2188 s | 39.8281 s | -11.92% |
| chord cProfile self time | about 12.08 s | about 2.48 s | descriptive only |

The retained-change gate was the larger of 5% and three baseline MAD. The
baseline MAD was 0.0256 s, so the observed 5.5867 s end-to-end reduction passes
the gate by a wide margin. The solver-owned memory plan is unchanged. The
warmed-process peak RSS maximum increased from 371,474,432 to 373,493,760
bytes (0.54%); this includes JIT code and is recorded rather than hidden.

The exact values, source hashes, and claim limits are in
[`metrics.json`](metrics.json). The full local raw rerun is
`.artifacts/m3c2-casep-after-chord-isolated-v1.json` with SHA-256
`db952c2a35527b32f06787126472327aed96d6b79246cae2a913075dfc1c641d`.
The separate profile report is retained locally as
`.artifacts/m3c2-casep-after-chord-v1.json`.

## Stop decision

This result authorizes only the retained chord consolidation. It does not
authorize a broad-phase framework, another runtime, internal threading, or a
sequence of hotspot rewrites. Product-scale 10,000/100,000/1,000,000-particle
throughput remains a separate SLA-backed task. The present work package stops
here because the one bounded change exceeded its gate while preserving the
accepted result exactly.
