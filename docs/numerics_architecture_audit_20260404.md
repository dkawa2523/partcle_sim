# Numerics / Boundary / Architecture Audit

Date: 2026-04-04

## Scope

This note summarizes the current numerical-processing specification and the main improvement opportunities across:

- time integration / difference methods
- boundary recognition / wall collision / geometry queries
- runtime configuration / architecture / operations

Reviewed files include:

- `particle_tracer_unified/solvers/high_fidelity_common.py`
- `particle_tracer_unified/solvers/integrator_common.py`
- `particle_tracer_unified/solvers/kernel2d_numba.py`
- `particle_tracer_unified/solvers/kernel3d_numba.py`
- `particle_tracer_unified/core/geometry2d.py`
- `particle_tracer_unified/core/geometry3d.py`
- `particle_tracer_unified/core/catalogs.py`
- `particle_tracer_unified/core/integrator_registry.py`
- `particle_tracer_unified/io/runtime_builder.py`
- `particle_tracer_unified/providers/precomputed.py`
- `particle_tracer_unified/solvers/source_preprocess.py`
- `particle_tracer_unified/solvers/solver2d.py`
- `particle_tracer_unified/solvers/solver3d.py`
- `tests/regression_test.py`
- example YAML files under `examples/`

## Current Specification

### Execution path

- YAML is parsed into a `RuntimeLike` in `build_runtime_from_config(...)`. Geometry, field, gas, wall catalog, physics catalog, particles, process steps, and compiled source events are assembled there. `particle_tracer_unified/io/runtime_builder.py:61`
- `prepare_runtime(...)` optionally preprocesses particles through source-material/source-event logic before the solver runs. `particle_tracer_unified/io/runtime_builder.py:158`
- `run_prepared_runtime(...)` owns the full step loop, particle release logic, free-flight integration, boundary detection, wall interaction, diagnostics, and output writing. `particle_tracer_unified/solvers/high_fidelity_common.py:1630`

### Physics / integrator model

- Public integrator names are `drag_relaxation`, `etd`, and `etd2`. `particle_tracer_unified/core/integrator_registry.py:7`
- The low-level component update solves a relaxation model per component with `tau_eff`, flow target, and body acceleration. `particle_tracer_unified/solvers/integrator_common.py:39`
- `drag_relaxation` uses a backward-compatible first-order position update, while `etd` and `etd2` share the same low-level exponential velocity update in `integrator_common.py`. `particle_tracer_unified/solvers/integrator_common.py:39`
- The actual ETD2 behavior is assembled in the Numba kernels by sampling midpoint flow and recording midpoint positions for collision broad-phase. `particle_tracer_unified/solvers/kernel2d_numba.py:116` `particle_tracer_unified/solvers/kernel3d_numba.py:129`
- Adaptive substepping is driven only by `dt / tau_eff`. `particle_tracer_unified/solvers/integrator_common.py:14`

### Boundary recognition and wall interaction

- 2D geometry truth uses boundary loops when available; otherwise it falls back to SDF sampling. `particle_tracer_unified/solvers/high_fidelity_common.py:92`
- 3D geometry truth uses triangle surfaces when available; otherwise it falls back to SDF sampling. `particle_tracer_unified/solvers/high_fidelity_common.py:126`
- 2D loop membership is computed by parity ray casting plus on-edge promotion. `particle_tracer_unified/core/geometry2d.py:65`
- 3D inside/outside is computed via nearest-surface tolerance plus a ray-casting parity test over triangles. `particle_tracer_unified/core/geometry3d.py:380`
- Collision handling is event-driven at the segment level: broad-phase hit detection, physical hit-time localization by bisection, then wall interaction and residual-time continuation. `particle_tracer_unified/solvers/high_fidelity_common.py:1225` `particle_tracer_unified/solvers/high_fidelity_common.py:1385`
- Wall behavior is resolved from `part_walls.csv` / `materials.csv` / step overrides and then applied as specular, diffuse, mixed, stick, absorb, or critical-sticking logic. `particle_tracer_unified/core/catalogs.py:183` `particle_tracer_unified/solvers/high_fidelity_common.py:626`

### Diagnostics and outputs

- The solver emits `wall_events.csv`, `wall_summary.json`, `wall_summary_by_part.csv`, `max_hit_events.csv`, and `collision_diagnostics.json`. `particle_tracer_unified/solvers/high_fidelity_common.py:2047` `particle_tracer_unified/solvers/high_fidelity_common.py:2061` `particle_tracer_unified/solvers/high_fidelity_common.py:2073` `particle_tracer_unified/solvers/high_fidelity_common.py:2099`
- Regression coverage is strong around collision diagnostics, inside-truth checks, retry-local adaptive behavior, and simple wall-reflection correctness. `tests/regression_test.py:846` `tests/regression_test.py:986` `tests/regression_test.py:1134` `tests/regression_test.py:1383` `tests/regression_test.py:1612`

## Findings

### P0. Integrator contract is ambiguous

- `integrator_common.py` treats `etd` and `etd2` identically at the low-level update stage. `particle_tracer_unified/solvers/integrator_common.py:39`
- ETD2-specific behavior actually lives in `kernel2d_numba.py`, `kernel3d_numba.py`, and the collision broad-phase in `high_fidelity_common.py`. `particle_tracer_unified/solvers/kernel2d_numba.py:116` `particle_tracer_unified/solvers/kernel3d_numba.py:129` `particle_tracer_unified/solvers/high_fidelity_common.py:1385`
- Result: the public integrator name does not map cleanly to one implementation unit. Order, dense-output behavior, and collision-stage semantics are split across three layers.

Recommended change:

- Introduce an internal integrator capability object or registry with explicit metadata:
- `order`
- `uses_midpoint_stage`
- `supports_dense_partial`
- `broad_phase_stage_count`
- Move ETD2-specific dense-output and midpoint logic behind one internal API so the public name and internal semantics stay aligned.

### P0. Boundary-query contract is not unified across 2D and 3D

- 2D uses loop parity with optional on-edge tolerance. `particle_tracer_unified/core/geometry2d.py:65`
- 3D uses nearest-triangle tolerance plus triangle parity ray casting. `particle_tracer_unified/core/geometry3d.py:380`
- Normals come from different paths: analytic edge normal orientation in 2D and triangle normal / nearest-surface projection in 3D. `particle_tracer_unified/solvers/high_fidelity_common.py:179` `particle_tracer_unified/core/geometry3d.py:344`
- Result: "inside", "on boundary", "nearest hit", and "nearest projection" are conceptually the same operations but are implemented with different APIs and different tolerance semantics.

Recommended change:

- Define one internal boundary-query interface for both 2D and 3D:
- `inside(position, tol)`
- `inside_strict(position)`
- `segment_hit(p0, p1)`
- `polyline_hit(p0, stage_points)`
- `nearest_projection(point, inside_reference)`
- Keep geometry-specific implementations behind that interface.

### P0. Step-level wall override semantics are fragile

- `resolve_step_wall_model(...)` uses numeric sentinel values to mean "inherit" for restitution and diffuse fraction. `particle_tracer_unified/core/catalogs.py:191`
- In practice, `1.0` and `0.0` are both valid user values and also act as "inherit" sentinels.
- Result: the configuration model is harder to reason about, and explicit user intent is not separable from inheritance.

Recommended change:

- Replace sentinel-based inheritance with explicit nullable fields or explicit `"inherit"` flags.
- Carry `None`/`NaN` through parsing and resolve defaults only once in the catalog layer.

### P1. 2D loop reconstruction is topology-light

- `build_boundary_loops_2d(...)` reconstructs loops by following arbitrary adjacency, with no explicit treatment of holes, nesting depth, or part segmentation. `particle_tracer_unified/core/geometry2d.py:8`
- `points_inside_boundary_loops_2d_with_boundary(...)` then applies XOR parity across the reconstructed loops. `particle_tracer_unified/core/geometry2d.py:65`
- Result: it works for the current closed-loop cases covered by tests, but complex multi-region or hole-rich geometries are under-specified.

Recommended change:

- Make loop building topology-aware:
- preserve per-loop orientation intentionally
- classify outer loops and holes by signed area and nesting
- keep part IDs attached to loop/edge groups
- Add regression cases for nested loops and hole geometries.

### P1. `valid_mask` is loaded and aligned but not used in the solver hot path

- Precomputed geometry and field load `valid_mask`. `particle_tracer_unified/providers/precomputed.py:71` `particle_tracer_unified/providers/precomputed.py:138`
- Runtime builder intersects field and geometry masks. `particle_tracer_unified/io/runtime_builder.py:49`
- The free-flight kernels sample by axis-clamped interpolation and do not consume `valid_mask`. `particle_tracer_unified/solvers/kernel2d_numba.py:11` `particle_tracer_unified/solvers/kernel3d_numba.py:11`
- Result: invalid regions are represented in data but ignored by sampling logic. This is a numerical-consistency gap, especially for sparse or cut-cell field bundles.

Recommended change:

- Add mask-aware sampling policy:
- reject samples outside valid cells
- optionally project to nearest valid cell
- or expose explicit extrapolation mode in config
- At minimum, surface a diagnostic when trajectories sample outside the valid mask.

### P1. Adaptive substepping is too narrow

- `compute_substep_count(...)` depends only on `dt`, `tau_eff`, and fixed split caps. `particle_tracer_unified/solvers/integrator_common.py:14`
- Boundary distance, flow curvature, and local interpolation error are ignored.
- Result: it controls stiffness crudely, but not geometry-driven or interpolation-driven error.

Recommended change:

- Add one of the following, in order of implementation cost:
- boundary-distance-aware step limiting
- flow-gradient-aware step limiting
- embedded local error estimate for ETD/ETD2 free flight
- Keep current retry-local adaptive mode as an operations fallback, not the main accuracy controller.

### P1. `run_prepared_runtime(...)` is oversized and multi-responsibility

- One function handles config extraction, initialization, step control, free-flight dispatch, collision handling, diagnostics, and file output. `particle_tracer_unified/solvers/high_fidelity_common.py:1630`
- Result: behavioral coupling is high, testing is harder, and design changes in one subsystem tend to leak into others.

Recommended change:

- Split `run_prepared_runtime(...)` into internal units:
- runtime options parsing
- per-step integrator advance
- collider handling
- diagnostics accumulation
- output serialization

### P1. 2D and 3D kernels duplicate most interpolation and stepping structure

- `_locate`, time interpolation, substep loops, and midpoint bookkeeping are duplicated between the 2D and 3D kernels. `particle_tracer_unified/solvers/kernel2d_numba.py:8` `particle_tracer_unified/solvers/kernel3d_numba.py:8`
- Result: changes in one kernel can drift from the other, and numerical fixes cost double.

Recommended change:

- Factor dimension-independent stepping logic into shared Numba helpers.
- Keep only dimension-specific interpolation and state packing/unpacking in per-dimension files.

### P2. Wrapper and preprocessing duplication remains

- `solver2d.py` and `solver3d.py` are near-identical wrappers. `particle_tracer_unified/solvers/solver2d.py:11` `particle_tracer_unified/solvers/solver3d.py:11`
- Source events are compiled in `runtime_builder.py` and checked/recompiled defensively again in `source_preprocess.py`. `particle_tracer_unified/io/runtime_builder.py:92` `particle_tracer_unified/solvers/source_preprocess.py:36`
- Result: code remains understandable, but there is unnecessary duplication and policy spread.

Recommended change:

- Unify solver wrappers behind a single dimension-aware entry point.
- Compile source events exactly once and make that ownership explicit.

### P2. Evaluation tooling is fragmented

- README documents runtime/unresolved-crossing gating via `tools/check_stability_profile.py`. `README.md:54` `README.md:76` `tools/check_stability_profile.py:40`
- `class_match_ratio` workflows exist in `demo_output` artifacts but not as a clear, canonical tool under `tools/`.
- Result: important quality metrics exist, but repeatable evaluation still depends on ad hoc scripts and prior output directories.

Recommended change:

- Add one supported evaluation tool that:
- runs candidate/reference configs
- computes `class_match_ratio`
- extracts `unresolved_crossing_count`
- records runtime
- writes one summary JSON/CSV

### P2. Operational clutter is high

- The workspace includes generated outputs and cache artifacts under source-adjacent paths such as `demo_output`, `tests/_out_*`, and many `__pycache__` directories.
- Result: navigation and review are noisier than necessary, and it is harder to see the authoritative code paths and evaluation outputs.

Recommended change:

- Establish a cleanup policy for generated outputs and cache artifacts.
- Keep durable benchmark outputs in a dedicated, versioned results location or regenerate them from a single tool.

## 2D / 3D Gaps Still Not Fully Unified

- 2D uses reconstructed boundary loops; 3D requires validated closed triangle surfaces. `particle_tracer_unified/core/geometry2d.py:8` `particle_tracer_unified/core/geometry3d.py:46`
- 2D fallback can use loop bisection on geometry truth; 3D fallback depends on triangle-surface projection and point-in-surface. `particle_tracer_unified/solvers/high_fidelity_common.py:292` `particle_tracer_unified/solvers/high_fidelity_common.py:337`
- 2D boundary part IDs ride edges and loops; 3D boundary part IDs ride triangles. `particle_tracer_unified/providers/precomputed.py:88` `particle_tracer_unified/providers/precomputed.py:91`
- 2D examples mostly use simple box or COMSOL loop geometries; 3D examples rely on closed-surface validation and a smaller example set. `tests/regression_test.py:846` `tests/regression_test.py:1134`

Implication:

- The solver is operationally unified, but the geometry contract is still dimension-specific. Future features should target a common boundary-service layer instead of adding more special cases to the main solver loop.

## Testing Gaps

Missing or underrepresented cases:

- 2D geometries with holes and nested loops
- 2D multi-part boundaries with shared vertices
- 3D near-grazing ray intersections and sliver triangles
- explicit user override of `wall_restitution=1.0` and `wall_diffuse_fraction=0.0` versus inherit semantics
- solver behavior when field `valid_mask` excludes cells near the trajectory path
- reproducibility tests for `mixed_specular_diffuse` under fixed RNG seed
- evaluation-tool tests for `class_match_ratio` workflows

## Recommended Work Packages

### WP1. Boundary service extraction

- Build a common geometry-query interface for 2D and 3D.
- Move inside/hit/projection/tolerance policy out of `run_prepared_runtime(...)`.
- Add hole/nesting regression tests before further geometry complexity is introduced.

### WP2. Integrator contract cleanup

- Extend the integrator registry to include capability metadata, not just names.
- Move ETD2 stage logic behind a single internal abstraction.
- Prepare for higher-order or embedded-error methods without further branching the main solver loop.

### WP3. Wall-model config cleanup

- Replace sentinel-value inheritance with explicit nullable configuration.
- Resolve all wall/default precedence once in the catalog layer.
- Add tests for step-default, step-override, part-wall, and material fallback precedence.

### WP4. Runtime/ops restructuring

- Split `run_prepared_runtime(...)` into smaller internal units.
- Consolidate solver wrappers and source-event compilation ownership.
- Add one canonical evaluation script for `class_match_ratio`, unresolved crossings, and runtime.

### WP5. Data and output hygiene

- Decide whether `valid_mask` is advisory or authoritative. Then enforce that contract in kernels.
- Separate durable benchmark outputs from disposable run artifacts.
- Add cleanup/ignore conventions for caches and transient outputs.

## Bottom Line

The current solver is functionally stronger than the earlier state: event-driven wall handling, dense partial free-flight replay, retry-local adaptive splitting, and inside-truth regressions are all in place.

The next quality bottlenecks are not one-off bug fixes. They are interface and contract problems:

- integrator semantics are split across layers
- boundary semantics are split across dimensions
- configuration inheritance mixes real values with sentinel values
- evaluation workflows are not yet a first-class tool

Those are the highest-leverage cleanup targets if the goal is to improve both accuracy and maintainability without reintroducing complexity.
