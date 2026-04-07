# Numerics Contract

## Continuous Model

The solver advances particles under a relaxation model

`dx/dt = v`

`dv/dt = (u(x, t) - v) / tau_p + a_body`

with:

- scalar particle relaxation time `tau_p`
- rectilinear precomputed, 2D triangle-mesh precomputed, or synthetic flow field sampling
- step-wise process controls for flow scale, drag-time scaling, and body-acceleration scaling

## Field Backend Contract

- `precomputed_npz` remains the rectilinear compatibility path.
- `precomputed_triangle_mesh_npz` is a 2D triangle-mesh backend.
- The mesh backend uses:
  - triangle containment as field support
  - barycentric interpolation on per-vertex quantities
  - a uniform candidate grid for triangle lookup acceleration
- The runtime loop and collision loop do not branch on field backend directly; backend differences are resolved in provider loading, shared sampling, and free-flight compile/kernels.

## Public Integrators

The public `solver.integrator` values remain:

- `drag_relaxation`
- `etd`
- `etd2`

Internal semantics are defined by `IntegratorSpec` in `particle_tracer_unified/core/integrator_registry.py`.

## Discrete Update Meaning

`drag_relaxation`

- first-order position update
- backward-compatible legacy mode
- one stage point per segment

`etd`

- exponential velocity update with frozen target flow over each substep
- one stage point per segment
- partial replay is supported by rerunning the same segment logic over a shorter `dt`

`etd2`

- midpoint-sampled exponential update
- midpoint is used for stage bookkeeping and collision broad-phase
- stage points are `[midpoint, endpoint]`
- partial replay reuses the same midpoint construction on the shorter segment

## Collision / Partial Replay Contract

- Free-flight integration for collision-aware replay goes through `advance_freeflight_segment(...)`.
- Dense partial replay for hit-time localization goes through `advance_freeflight_partial(...)`.
- Wall reflection is evaluated from the hit-time state `(x_hit, v_hit)`, not from the segment endpoint state.
- ETD2 midpoint data is used for broad-phase geometry checks only; post-hit continuation is always recomputed from the physical hit state.

## `valid_mask` Contract In This Tranche

- `valid_mask` is authoritative in source preprocessing samplers.
- For precomputed fields, load-time regularization is applied before runtime sampling:
  - `core_valid_mask = field.valid_mask & geometry.valid_mask`
  - `effective_valid_mask = core_valid_mask + geometry-aware narrow band`
  - band width is one max-cell diagonal based on the rectilinear grid
  - quantity values on the added band are filled from inward-normal donor samples, with nearest core-valid node fallback
- Runtime sampling and diagnostics use the regularized `effective_valid_mask`.
- The solver hot path supports two policies:
  - `diagnostic` (default): current behavior, diagnostics only
  - `retry_then_stop` (opt-in): invalid trial steps are retried on shorter prefixes and then stopped at the last valid state
- Kernel and replay paths classify `valid_mask` samples into:
  - `clean`
  - `mixed_stencil`: sampled point is still valid, but the interpolation stencil touches invalid cells
  - `hard_invalid`: the sampled point itself is invalid
- Under `retry_then_stop`, the same halving-based prefix resolver is used for both:
  - the initial per-step free-flight trial
  - collision-replay segments after a wall hit
- Under `retry_then_stop`, only `hard_invalid` segments trigger prefix retry and stop behavior.
- `mixed_stencil` remains diagnostic-only so near-wall mixed cells do not become terminal stops by themselves.
- If collision replay enters a `hard_invalid` region, the particle stops at the best valid prefix on that local segment.
- If no valid prefix is found within the configured halving budget, the particle stays at the segment-start state for that replay segment.
- Aggregate diagnostics remain backward-compatible:
  - `valid_mask_violation_count = mixed_stencil + hard_invalid`
  - split counts are reported separately for mixed and hard-invalid cases
- Extension-band usage is reported separately so rollout evaluation can distinguish:
  - samples that needed the added support band
  - samples that still became `hard_invalid` even after regularization
- Precomputed-field regularization metadata reports:
  - mode
  - band distance
  - added node count
  - inward-probe success count
  - nearest-core fallback count
- `retry_then_stop` is rollout-only for now; it is not the default solver behavior.

## Current Geometry Truth Contract

- 2D truth source is `boundary_loops_2d` when available, otherwise SDF fallback.
- 3D truth source is a validated closed triangle surface when available, otherwise SDF fallback.
- `BoundaryService` is the internal contract for `inside`, `inside_strict`, `segment_hit`, `polyline_hit`, and `nearest_projection`.
- 2D loop truth uses even-odd parity across disjoint loops, so nested-hole cases covered by regression tests are supported.
- 2D boundary-edge inputs are required to form disjoint degree-2 loops; shared-vertex branching or dangling topologies are rejected early instead of being silently reconstructed.
