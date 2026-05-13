# Architecture

This project is easiest to read from the runtime boundary inward.

## Runtime Flow

1. `run_from_yaml.py` or `particle-tracer-run` reads a case config.
2. `particle_tracer_unified/io/runtime_builder.py` builds a `RuntimeLike`.
3. `prepare_runtime` optionally applies source preprocessing.
4. solver entry points consume `PreparedRuntime`.
5. output modules write summaries and diagnostics.

`runtime_builder.py` should stay a small coordinator. Format-specific loading
belongs in focused `io/` modules.

## Package Map

- `particle_tracer_unified/io/`: CSV/NPZ/YAML loading and runtime construction
- `particle_tracer_unified/io/comsol.py`: COMSOL faithful runtime integration
- `particle_tracer_unified/core/`: shared data models, geometry, source events,
  boundary services, and catalogs
- `particle_tracer_unified/providers/`: geometry and field providers
- `particle_tracer_unified/solvers/`: forces, integration, collisions, and
  runtime outputs
- `particle_tracer_unified/compare/`: field, acceleration, trajectory, and
  boundary comparison CLIs
- `tools/`: repository utilities and case builders
- `external/`: tooling that talks to outside systems such as COMSOL
- `examples/`: runnable or template cases

## Extension Rules

- Add new data formats in `io/`, then pass normalized tables/providers into the
  existing runtime.
- Add field or geometry behavior behind provider interfaces rather than inside
  the solver loop.
- Add solver behavior behind `PreparedRuntime` and existing catalogs.
- Keep COMSOL API calls outside `particle_tracer_unified`; the solver package
  should consume exported CSV/NPZ/YAML data.
- Prefer one validation point near data loading. Avoid repeating the same check
  in every downstream layer.
- Keep tests focused on behavior that can regress. Avoid large fixture stacks
  when a small table or synthetic provider proves the same contract.

## Cleanup Bias

When improving code, remove obsolete comments, copied specs, and dead branches
before adding new abstractions. A helper is worth keeping only when it hides a
real repeated pattern or gives a third-party reader a clearer name for the
operation.
