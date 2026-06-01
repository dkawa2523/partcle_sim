# Skill 00: Global Rules for COMSOL Comparison Work

## Purpose

COMSOLファイルやexport資産を使って本コードの粒子軌道結果と比較する際に、Codexが守るべき全体ルールを定義する。

## Non-negotiable rules

1. Do not tune final endpoint counts only.
2. Do not add VIGUS-specific wall IDs, part IDs, source IDs, or geometry names to solver core.
3. Do not treat a COMSOL handoff package as a runnable solver case unless preflight proves it.
4. Do not silently snap, repair, extrapolate, or fill missing COMSOL data in `comsol_faithful` mode.
5. Do not confuse production surface release preprocessing with COMSOL faithful release comparison.
6. Do not add broad contract frameworks, helper layers, or always-on diagnostics.
7. Prefer compact artifacts and requested-only diagnostics.
8. Any stochastic comparison must record seeds and distinguish deterministic vs ensemble validation.
9. When COMSOL metadata is ambiguous, create an extraction gap report and stop rather than guessing.
10. Keep each code change tied to a specific comparison layer.

## Correct comparison order

Always compare in this order:

```text
L0 import / manifest
L1 geometry, coordinate, units
L2 field sampling
L3 release table and preprocessing
L4 first-step force / integrator
L5 wall event / boundary hit
L6 trajectory / first passage / ensemble
L7 runtime / performance
```

A later layer is not meaningful if an earlier layer fails.

## Required stop conditions

Stop and report instead of patching if any of these are true:

- COMSOL model component/study/dataset is unknown.
- Coordinate system is unknown or inconsistent.
- Unit scale is unknown.
- Field export does not cover released particles.
- Wall law mapping is missing for selected boundaries.
- Release table cannot identify particle IDs, times, positions, and velocities.
- Brownian/stochastic terms are enabled but seed or ensemble protocol is unknown.
- COMSOL reference data mixes sampled and full particles without labels.

## Minimal output discipline

Default outputs should be compact:

```text
summary JSON
small CSV for mismatches
short Markdown report
```

Avoid dumping full trajectories, wall-event logs, or per-step diagnostics unless the current skill explicitly requests them.

## What to write at the end of every Codex task

```text
Changed files:
Behavior changed:
Comparison layer affected:
Tests/checks run:
Artifacts generated:
Deferred issues:
```
