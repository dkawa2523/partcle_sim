# Numerics / Architecture Review

Date: 2026-04-04

Scope:
- `particle_tracer_unified/io/runtime_builder.py`
- `particle_tracer_unified/io/tables.py`
- `particle_tracer_unified/core/datamodel.py`
- `particle_tracer_unified/core/process_steps.py`
- `particle_tracer_unified/core/catalogs.py`
- `particle_tracer_unified/core/source_events.py`
- `particle_tracer_unified/solvers/source_preprocess.py`
- `particle_tracer_unified/solvers/integrator_common.py`
- `particle_tracer_unified/solvers/kernel2d_numba.py`
- `particle_tracer_unified/solvers/kernel3d_numba.py`
- `particle_tracer_unified/solvers/high_fidelity_common.py`
- `particle_tracer_unified/solvers/solver2d.py`
- `particle_tracer_unified/solvers/solver3d.py`
- `run_from_yaml.py`
- `README.md`
- `tests/regression_test.py`
- `tests/smoke_test.py`

## 1. Current Code Specification

### 1.1 Configuration and runtime build flow

The current runtime construction path is:

`run_config.yaml`
-> `build_runtime_from_config(...)`
-> CSV/YAML table loaders
-> provider builders
-> wall/physics catalogs
-> source-event compilation
-> source preprocessing
-> `PreparedRuntime`
-> solver entrypoint
-> `run_prepared_runtime(...)`

Relevant code:
- `particle_tracer_unified/io/runtime_builder.py:61`
- `particle_tracer_unified/io/runtime_builder.py:166`
- `particle_tracer_unified/io/runtime_builder.py:197`
- `run_from_yaml.py:10`
- `particle_tracer_unified/solvers/solver2d.py:11`
- `particle_tracer_unified/solvers/solver3d.py:11`

Configuration is split across:
- top-level YAML
- `particles.csv`
- `materials.csv`
- `part_walls.csv`
- `process_steps.csv` or `recipe_manifest.yaml`
- `source_events.csv`
- process `step_defaults`
- process `step_overrides`

The effective runtime state is therefore a merged product of file inputs and YAML overlays rather than a single explicit schema document.

### 1.2 Source preprocessing

Particle source/material/event parameters are resolved before the solver runs. Preprocessing mutates the particle table and returns a `SourcePreprocessResult`.

Relevant code:
- `particle_tracer_unified/solvers/source_preprocess.py:11`
- `particle_tracer_unified/core/source_materials.py`
- `particle_tracer_unified/core/source_events.py:109`

### 1.3 Solver structure

The solver entrypoints are thin wrappers. Almost all actual solver responsibilities are centralized in `high_fidelity_common.py`.

Relevant code:
- `particle_tracer_unified/solvers/solver2d.py:22`
- `particle_tracer_unified/solvers/solver3d.py:22`
- `particle_tracer_unified/solvers/high_fidelity_common.py:1630`

Current responsibilities concentrated in `high_fidelity_common.py`:
- solver config normalization
- per-step process control resolution
- free-flight kernel dispatch
- boundary detection and reintegration
- wall interaction
- diagnostics accumulation
- CSV/JSON/NPY output writing
- trajectory plot generation

File size context:
- `high_fidelity_common.py`: 2160 lines
- `source_materials.py`: 698 lines
- `tables.py`: 465 lines
- `datamodel.py`: 487 lines
- `regression_test.py`: 1798 lines

### 1.4 Numerical core

Free-flight uses Numba kernels plus low-level ETD / drag updates:
- `integrator_common.py` provides low-level component/state updates and heuristic substep counts.
- `kernel2d_numba.py` and `kernel3d_numba.py` perform interpolation and particle stepping.
- `high_fidelity_common.py` handles collision-aware segment reintegration around those kernels.

Relevant code:
- `particle_tracer_unified/solvers/integrator_common.py:13`
- `particle_tracer_unified/solvers/kernel2d_numba.py:53`
- `particle_tracer_unified/solvers/kernel3d_numba.py:63`
- `particle_tracer_unified/solvers/high_fidelity_common.py:1037`
- `particle_tracer_unified/solvers/high_fidelity_common.py:1385`

### 1.5 Output contract

A normal run writes many artifacts by default:
- `final_particles.csv`
- `wall_events.csv`
- `wall_summary.json`
- `wall_summary_by_part.csv`
- `max_hit_events.csv`
- `runtime_step_summary.csv`
- `collision_diagnostics.json`
- `prepared_runtime_summary.json`
- `positions_*.npy`
- segmented position arrays
- source-preprocess summaries
- a trajectory PNG

Relevant code:
- `particle_tracer_unified/solvers/high_fidelity_common.py:2045`
- `particle_tracer_unified/solvers/high_fidelity_common.py:2077`
- `particle_tracer_unified/solvers/high_fidelity_common.py:2099`
- `particle_tracer_unified/solvers/high_fidelity_common.py:2133`
- `particle_tracer_unified/solvers/high_fidelity_common.py:2136`

## 2. Findings: Integrator / Difference Method Area

### P0. Numerical assumptions are not documented at the code boundary

The solver currently assumes:
- drag relaxation with scalar `tau_p`
- flow interpolation from rectilinear precomputed or synthetic fields
- ETD / ETD2 built around frozen or midpoint-sampled target velocity over each substep

Those assumptions are spread across `integrator_common.py`, Numba kernels, and `high_fidelity_common.py`, but there is no single explicit numerics-spec document or in-code contract.

Impact:
- hard to reason about what each integrator guarantees
- hard to extend without accidentally changing semantics

Relevant code:
- `particle_tracer_unified/solvers/integrator_common.py:27`
- `particle_tracer_unified/solvers/kernel2d_numba.py:111`
- `particle_tracer_unified/solvers/kernel3d_numba.py:123`

### P1. Adaptive substepping is a heuristic, not an error-controlled policy

`compute_substep_count(...)` uses only `dt / tau_eff` and `adaptive_substep_max_splits`.

It does not consider:
- local field gradient
- event proximity
- boundary distance
- embedded truncation error

Impact:
- behavior depends on tuning rather than a measurable error target
- solver cost can increase without a direct accuracy guarantee

Relevant code:
- `particle_tracer_unified/solvers/integrator_common.py:13`

### P1. ETD2 midpoint logic is duplicated in 2D and 3D kernels

`kernel2d_numba.py` and `kernel3d_numba.py` each implement the same ETD2 midpoint/substep pattern independently.

Impact:
- any future integrator change must be mirrored twice
- easy to drift between 2D and 3D behavior

Relevant code:
- `particle_tracer_unified/solvers/kernel2d_numba.py:111`
- `particle_tracer_unified/solvers/kernel3d_numba.py:123`

### P1. Free-flight reference / verification mode is externalized rather than first-class

The repository contains external evaluation workflows and historical outputs, but the solver itself has no built-in high-accuracy reference mode, tolerance target, or comparison harness.

Impact:
- regression evaluation depends on custom scripts or ad hoc runs
- accuracy drift can be harder to detect during routine development

Evidence:
- `README.md:59`
- `tests/regression_test.py:184`
- `demo_output/*integrator_eval*`

### Recommended improvements

1. Add a short numerics-spec document that defines the continuous model, the discrete update actually implemented by each integrator, and the intended accuracy/stability envelope.
2. Replace heuristic-only adaptive stepping with a two-layer policy:
   - fast default heuristic
   - optional error-controlled mode for verification and difficult cases
3. Extract a shared ETD2 substep engine used by both 2D and 3D kernels.
4. Add a built-in comparison harness for `coarse vs reference` runs under fixed seed and fixed configs.

## 3. Findings: Boundary Conditions / Wall Interaction Area

### P0. Boundary semantics are implemented, but not codified as a single truth ladder

Boundary truth is currently chosen implicitly from available geometry data:
- 2D loop truth
- 2D boundary edges
- 3D triangle surface
- SDF fallback

The precedence exists in code but is not documented as an explicit policy.

Impact:
- difficult to understand which geometry source is authoritative
- behavior can change depending on provider contents

Relevant code:
- `particle_tracer_unified/core/geometry2d.py`
- `particle_tracer_unified/core/geometry3d.py`
- `particle_tracer_unified/solvers/high_fidelity_common.py:93`
- `particle_tracer_unified/solvers/high_fidelity_common.py:127`

### P0. Step-aware wall configuration still uses sentinel-based inheritance

`process_steps.py` and `catalogs.py` use literal values like `1.0`, `0.0`, and `'inherit'` to mean both "explicit value" and "fallback to base" depending on context.

Examples:
- `wall_restitution == 1.0`
- `wall_diffuse_fraction == 0.0`
- `wall_stick_probability_scale == 1.0`

Impact:
- explicit values can be indistinguishable from "not overridden"
- operational tuning becomes ambiguous
- boundary-condition experiments become harder to trust

Relevant code:
- `particle_tracer_unified/core/process_steps.py:34`
- `particle_tracer_unified/core/process_steps.py:77`
- `particle_tracer_unified/core/catalogs.py:183`

This is not just a documentation issue. It is a semantics issue:
- an explicit step value equal to the sentinel can be collapsed into "inherit"
- review of a config file is therefore insufficient to know the actual effective wall setup

### P1. 2D and 3D collision classification remain structurally asymmetric

The recent commonization improved the colliding-particle path, but broad-phase classification still differs:
- 2D uses vectorized loop-inside tests
- 3D uses per-particle branching with triangle hits and bbox checks

Impact:
- different failure modes and tuning behavior between 2D and 3D
- higher maintenance cost

Relevant code:
- `particle_tracer_unified/solvers/high_fidelity_common.py:1824`
- `particle_tracer_unified/solvers/high_fidelity_common.py:1849`

### P1. Boundary diagnostics are strong, but still underspecified for deep debugging

Current diagnostics expose counts, outcomes, and wall modes. They do not persist the richer event-state data that would help reproduce difficult cases:
- hit time
- remaining time before/after hit
- fallback path used
- local retry/adaptive status per event

Impact:
- difficult to postmortem a single bad particle without rerunning under instrumentation

Relevant code:
- `particle_tracer_unified/solvers/high_fidelity_common.py:1726`
- `particle_tracer_unified/solvers/high_fidelity_common.py:2047`
- `particle_tracer_unified/solvers/high_fidelity_common.py:2073`

### Recommended improvements

1. Add a documented "geometry truth ladder" and require each provider to declare which boundary truth objects it supplies.
2. Replace sentinel-based wall/step inheritance with `Optional` / `NaN`-backed unresolved state until final merge time.
3. Continue the 2D/3D commonization by extracting a boundary-query interface with a unified broad-phase API.
4. Add optional rich wall-event logging for targeted debugging runs.

## 4. Findings: Operations / Architecture / Maintainability

### P0. Solver responsibilities are too concentrated in `high_fidelity_common.py`

`run_prepared_runtime(...)` owns:
- config parsing
- step iteration
- kernel dispatch
- collision logic
- diagnostics
- output generation
- plotting

Impact:
- difficult to review safely
- difficult to unit test in isolation
- high coupling between numerics, I/O, and visualization

Relevant code:
- `particle_tracer_unified/solvers/high_fidelity_common.py:1630`

### P0. Configuration precedence is powerful but opaque

Effective behavior is derived from multiple merge layers:
- table defaults
- YAML defaults
- process step rows
- step defaults
- step overrides
- part/material wall catalogs
- source preprocessing results

There is no single precedence table in docs or code comments.

Impact:
- operator confusion
- fragile bug triage
- configuration reviews require reading several files and functions together

Relevant code:
- `particle_tracer_unified/io/runtime_builder.py:61`
- `particle_tracer_unified/io/tables.py:345`
- `particle_tracer_unified/io/tables.py:394`
- `particle_tracer_unified/core/process_steps.py:77`
- `particle_tracer_unified/core/catalogs.py:183`

### P1. Source events are compiled twice on the hot path into `PreparedRuntime`

`build_runtime_from_config(...)` compiles source events. `preprocess_particles_for_solver(...)` may compile them again unless metadata indicates they are already compiled.

Impact:
- redundant work
- duplicated state logic
- added cognitive load for event lifecycle

Relevant code:
- `particle_tracer_unified/io/runtime_builder.py:92`
- `particle_tracer_unified/solvers/source_preprocess.py:36`

### P1. Input validation is permissive enough to hide bad data

Examples:
- `load_particles_csv(...)` silently defaults many columns
- process steps are sorted but not strongly validated for overlap/gap consistency
- `ProcessStepTable.active_at(...)` extrapolates to first/last step outside defined ranges

Impact:
- malformed or partial inputs may produce plausible but unintended runs
- production debugging becomes harder

Relevant code:
- `particle_tracer_unified/io/tables.py:74`
- `particle_tracer_unified/io/tables.py:345`
- `particle_tracer_unified/io/tables.py:394`
- `particle_tracer_unified/core/datamodel.py:289`

### P1. Output writing is heavy and only partially configurable

A normal run always writes:
- large NPY trajectory arrays
- segmented NPY outputs
- multiple CSV/JSON summaries
- trajectory plots

Only some diagnostics are configurable. Plotting and most report generation are unconditional.

Impact:
- unnecessary I/O for batch runs
- slower CI / production runs
- solver core is coupled to reporting choices

Relevant code:
- `particle_tracer_unified/solvers/high_fidelity_common.py:2045`
- `particle_tracer_unified/solvers/high_fidelity_common.py:2077`
- `particle_tracer_unified/solvers/high_fidelity_common.py:2136`

There is also output-path duplication:
- `run_from_yaml.py --prepare-only` writes `prepared_runtime_summary.json` directly
- normal solver runs write the same summary again inside `high_fidelity_common.py`

Relevant code:
- `run_from_yaml.py:24`
- `particle_tracer_unified/solvers/high_fidelity_common.py:2077`

### P1. 2D and 3D wrapper modules are near-duplicates

`solver2d.py` and `solver3d.py` differ mainly by spatial-dimension guard and dispatch.

Impact:
- low-value duplication
- more places to touch for behavior changes

Relevant code:
- `particle_tracer_unified/solvers/solver2d.py:11`
- `particle_tracer_unified/solvers/solver3d.py:11`

### P1. `replace_runtime_particles(...)` weakens type safety

When dataclass replacement fails, the function falls back to generic object copying and direct attribute mutation.

Impact:
- harder static reasoning
- more fragile extension path
- hidden runtime shape assumptions

Relevant code:
- `particle_tracer_unified/core/datamodel.py:468`

### P2. Flat data models are oversized and mix multiple domains

`MaterialRow` and `PartWallRow` carry:
- source parameters
- resuspension parameters
- wall parameters
- optional physics defaults

Impact:
- hard to understand ownership and precedence
- noisy constructors and loaders

Relevant code:
- `particle_tracer_unified/core/datamodel.py:104`
- `particle_tracer_unified/core/datamodel.py:165`
- `particle_tracer_unified/io/tables.py:120`
- `particle_tracer_unified/io/tables.py:193`

### Recommended improvements

1. Split `high_fidelity_common.py` into:
   - step runner
   - collision/boundary subsystem
   - output/report writer
   - plotting/visualization
2. Introduce a single normalized config model before runtime build.
3. Make source-event compilation single-pass and explicit.
4. Add strict input validation mode:
   - required-column checks
   - process-step overlap/gap validation
   - unknown override-name detection
5. Move output policy behind explicit flags or profiles:
   - solver-only
   - diagnostics
   - full visualization

## 5. Findings: Documentation / Test Coverage

### Gaps in documentation

Missing or under-documented topics:
- configuration precedence
- geometry/field compatibility requirements
- process-step inheritance semantics
- wall catalog precedence
- output artifact contract and cost
- numerics assumptions per integrator

Evidence:
- `README.md` documents example runs and the ETD2 stability profile, but not the full merge/precedence model.

### Gaps in tests

Current positives:
- good regression coverage for integrator names, ETD2 diagnostics, wall events, collision diagnostics, and geometry truth
- smoke coverage for minimal 2D/3D CLI path

Current gaps:
- no focused test suite for configuration precedence edge cases
- no strict validation tests for malformed process-step schedules
- no dedicated tests proving explicit `1.0` / `0.0` overrides survive merge semantics
- no isolated tests for output-policy toggles because most outputs are always written
- regression suite is concentrated into a single large file
- smoke tests write into repository-local output directories rather than ephemeral temp directories

Relevant code:
- `tests/regression_test.py:129`
- `tests/regression_test.py:184`
- `tests/regression_test.py:1134`
- `tests/smoke_test.py:8`

### Recommended test/doc improvements

1. Split `tests/regression_test.py` into themed modules:
   - config/runtime build
   - integrators
   - boundaries/collisions
   - output/reporting
2. Add a dedicated config-precedence test matrix.
3. Add tests for invalid step schedules and unknown step override names.
4. Add a short "configuration precedence" document and a "solver output contract" document.

## 6. Prioritized Improvement Backlog

### Phase 1: correctness and operational clarity

1. Remove sentinel-based inheritance for process-step and wall settings.
2. Add strict config validation and schedule validation.
3. Make source-event compilation single-pass.
4. Document configuration precedence and geometry truth precedence.

### Phase 2: architecture cleanup

1. Split `high_fidelity_common.py` by responsibility.
2. Unify `solver2d.py` / `solver3d.py` into a single generic entry wrapper.
3. Replace flat row types with grouped substructures or typed views.
4. Centralize output writing behind an output-policy layer.

### Phase 3: numerical and verification improvements

1. Add an internal reference-comparison harness.
2. Add error-aware adaptive stepping mode.
3. Expand per-event diagnostics for hard collision cases.
4. Continue unifying 2D/3D boundary-query interfaces.

## 7. Executive Summary

The codebase is functional and now materially stronger on collision consistency, but the next bottleneck is not only numerical. The main medium-term risk is that configuration semantics, solver semantics, and reporting semantics are still too entangled.

The highest-value next actions are:
- fix inheritance semantics
- split solver core from output/report generation
- validate configuration more strictly
- codify the precedence rules and geometry truth rules in documentation and tests

Without that cleanup, further numerical upgrades will continue to cost more than they should in review time, debugging time, and operator confusion.
