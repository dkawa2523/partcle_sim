# Read Before Implementation

This is not a prompt. It is the implementation context Codex should read before making changes.

## Core mission

Improve sim_rev3 so it can:
- ingest fields, geometry, settings, and particle initial conditions from COMSOL or other simulations;
- compute particle trajectories faster than COMSOL Particle Tracing for production exploration;
- preserve enough COMSOL parity to make comparisons meaningful;
- remain small, readable, and maintainable as product code.

## Critical shift in v2

Do not start by changing wall physics or release physics.

Start by making comparison safe:

```text
Phase 0   audit
Phase 0.5 focused checks
Phase 1   minimal compare rails
```

Only after that should mode separation, manifest, release, force, and wall changes proceed.

## Golden rules

1. Implement one phase at a time.
2. Keep diffs small.
3. Do not add VIGUS-specific wall IDs or source IDs to solver core.
4. Do not add broad frameworks.
5. Do not make diagnostics always-on.
6. Do not tune to one local metric.
7. Do not use old `_tmp_*` outputs as product truth.
8. Add minimal tests only for changed behavior.
9. At the end of each phase, record files changed and tests run.
10. If comparison is not available, do not claim physics improved.

## What to optimize

Correctness first, then speed.

Speed problems caused by wrong event semantics should be fixed as physics/event semantics, not as hot-loop micro-optimization.
