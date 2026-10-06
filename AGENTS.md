# Clean-room workspace guidance

This repository is being rebuilt from zero. Before changing anything, use the
following ownership rules.

## Active areas

- `particle_platform_redesign/` owns the new specification, architecture,
  implementation plan, and current solver. Read
  `particle_platform_redesign/AGENTS.md` in full before working there.
- `model_dataset/` is reference input and external V&V material. It is not a
  solver dependency or a golden definition of the physics.
- `old_code/` is a read-only archive of the former implementation, tools,
  tests, environment, and root files. Do not activate its environment, import
  its package, or modify it to implement the new solver.

## Clean-room rules

- New production code belongs only in the independent uv project at
  `particle_platform_redesign/solver/`.
- Do not restore an archived root `pyproject.toml`, `uv.lock`, package, test
  tree, or tool configuration.
- The old implementation may be inspected only for failure cases, candidate
  equations, and input provenance. Re-derive and independently verify anything
  adopted by the new solver.
- COMSOL integration and comparisons stay in adapters and external V&V tools;
  the solver core must not depend on COMSOL or `model_dataset/`.
- Keep one production engine and one owner for each fact. Replaced code,
  settings, tests, and documentation are removed in the same change.
- Do not add general plugin frameworks, dependency-injection containers,
  duplicated validators, broad contract suites, or diagnostic subsystems.

The detailed numerical invariants, module ownership, quality gates, and change
workflow are authoritative in `particle_platform_redesign/AGENTS.md`.

## Project skills

- Use `$prepare-comsol-field-case` to turn a COMSOL geometry/field model without
  particle tracing into a checked solver case.
- Use `$validate-comsol-trajectories` for a meaning-matched external comparison
  with a COMSOL particle-tracing model.
- Use `$visualize-particle-trajectories` for geometry-overlaid trajectories,
  animations, population views, and COMSOL comparison figures.
