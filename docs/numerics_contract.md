# Numerics Contract

## Continuous Model

The solver advances particles in supplied geometry, flow fields, electric fields, and gas-property fields under a relaxation model

`dx/dt = v`

`dv/dt = (u(x, t) - v) / tau_eff + a_body + (q(t) / m) E(x, t)`

with:

- particle relaxation time `tau_eff` from the selected drag model
- rectilinear precomputed, 2D triangle-mesh precomputed, or synthetic flow field sampling
- steady or time-dependent flow sampling through the field time axis
- optional provider-backed electric fields sampled as `E_x/E_y` or `E_r/E_z`; charged particles use the sampled electric field directly as `(q(t)/m)E`
- constant body acceleration, with optional lightweight step-wise scaling retained for compatibility

The default solver path is deterministic. The core solver does not infer missing fields, train surrogate models, or learn trajectory corrections. Analysis and optimization tooling must stay outside the trajectory hot path. Stochastic physics and charge evolution are explicit opt-in model behavior and must not act as field or boundary rescue logic.

`solver.drag_model` controls how the relaxation time is interpreted:

- `stokes` keeps the original linear Stokes relaxation model.
- `schiller_naumann` applies a finite-Re drag correction from the local slip speed and gas properties, reducing the effective relaxation time as particle Reynolds number rises.
- `epstein` uses low-pressure free-molecular relaxation. Rectilinear field bundles may provide `rho_g` and `T`; these are sampled at the same trajectory stage points as the flow field. `gas.density_kgm3` and `gas.temperature_K` remain scalar fallbacks when a field quantity is missing or invalid. Pressure `p` is diagnostic only and is not used directly by drag. Field `mu` can also be carried and reported, but it is not part of the Epstein relaxation formula.

`solver.charge_model` is disabled by default. In 2D regular rectilinear fields it supports:

- `te_relaxation` (`v1`): relaxes particle charge toward `q_eq = -4 pi eps0 r_p alpha Te`, where `Te` is in eV and is numerically the floating-potential scale in volts.
- `density_temperature_flux_relaxation` (`v2`): computes a local floating potential from electron/ion density and temperature using density-temperature flux balance, then relaxes `q` toward `4 pi eps0 r_p phi_f`.

Both modes sample only local scalar background distributions such as `Te`, `ne`, `ni`, and `Ti`. They do not consume COMSOL flux vectors. The acceleration term is then evaluated as `(q/m)E`, so time-dependent background distributions are handled through the normal field time axis. 3D dynamic charge coupling is intentionally left for a separate implementation.

## Field Backend Contract

- `precomputed_npz` remains the rectilinear compatibility path.
- `precomputed_triangle_mesh_npz` is a 2D triangle-mesh backend for true mesh field data; the COMSOL builder does not synthesize mesh fields from rectilinear bundles.
- `provider_contract.boundary_field_support` checks whether explicit boundary samples can be offset just inside the geometry and sampled cleanly by the field provider.
- Strict provider-contract failure stops before time integration and writes `provider_contract_report.json`, `provider_boundary_summary.csv`, and `provider_boundary_violations.csv`.
- `provider_contract_report.json` includes the field support summary, per-part violation bounding boxes, and every sampled non-clean boundary support point, so the report is diagnostic rather than a solver-side rescue path.
- `provider_contract_report.json` also includes `geometry_boundary` so users can confirm the explicit 2D loop or 3D triangle-surface truth used by the boundary check.
- `provider_boundary_summary.csv` gives the same per-part failure counts and boundary/offset bounding boxes in a compact table for export-bundle triage.
- `provider_boundary_violations.csv` uses numeric `boundary_*`, `offset_*`, and `checked_time_s` columns for direct plotting and export-bundle debugging.
- Transient provider checks evaluate representative field times through `provider_contract.max_time_samples`.
- `field_support.time_axis` reports the common field time axis and flags quantity-level time-axis mismatches. A strict provider contract treats such mismatches as an export/provider problem, not as a solver interpolation feature.
- Field support is provider-native. It is not clipped to the geometry mask during runtime assembly; the geometry/field intersection is kept only as a diagnostic `core_valid_mask`.
- The COMSOL builder exports rectilinear field support from finite field quantities and adds an edge-extrapolated ghost-cell band, defaulting to 8 cells. It does not reuse a geometry/domain mask as field support, because wall-adjacent interpolation may need finite ghost/support nodes outside the physical domain.
- Precomputed providers validate monotone finite axes/times, finite values on active support, and non-degenerate in-range mesh triangles before solver execution.
- Provider field bundles should not precompute electric acceleration from a fixed reference charge/mass.
- Electric acceleration is evaluated in the solver from the current particle state as `(q(t)/m)E`, so charge evolution and electric force cannot diverge.
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
- For precomputed rectilinear fields, `field.valid_mask` remains provider-native. The geometry intersection is retained as diagnostic `core_valid_mask`, not as the field provider's sampling support.
- The loader does not synthesize additional valid nodes or fill missing field values near boundaries. Export/build tools may add finite ghost/support nodes before the provider is loaded.
- Boundary-adjacent field support is a provider responsibility, not a solver rescue responsibility.
- Initial particles must be inside the clean field sample domain when `input_contract.initial_particle_field_support` is `strict`:
  - `clean`: the sampled point and interpolation stencil are valid
  - `mixed_stencil`: the point is valid, but interpolation touches invalid nodes
  - `hard_invalid`: the point itself is outside valid field support
- Strict input-contract failure stops before time integration and writes `input_contract_report.json` plus `input_particle_violations.csv`.
- Points outside the field axes are `hard_invalid`; they are not clamped to the nearest grid boundary for validity decisions.
- The solver hot path supports two policies:
  - `retry_then_stop` (default): hard-invalid trial segments are stopped before they can contaminate deposition/coating results
  - `diagnostic` (explicit investigation mode): diagnostics only
- Under `retry_then_stop`, hard-invalid trial segments are resolved by a halving-based valid-prefix retry. The stopped particle remains classified as `invalid_mask_stopped`; the solver does not convert this condition to deposition, sticking, or wall contact.
- Provider `support_phi` and geometry SDF may be reported as diagnostic field/geometry features, but they are not used as an active stop-location or collision-rule override in the solver path.
- Kernel and replay paths classify `valid_mask` samples into:
  - `clean`
  - `mixed_stencil`: sampled point is still valid, but the interpolation stencil touches invalid cells
  - `hard_invalid`: the sampled point itself is invalid
- Under `retry_then_stop`, the same valid-prefix behavior is used for both:
  - the initial per-step free-flight trial
  - collision-replay segments after a wall hit
- Under `retry_then_stop`, only `hard_invalid` segments trigger valid-prefix stop behavior.
- `mixed_stencil` remains diagnostic-only so near-wall mixed cells do not become terminal stops by themselves.
- If collision replay enters a `hard_invalid` region, the particle stops at the best valid prefix on that local segment.
- If no valid prefix is found within the configured halving budget, the particle stays at the segment-start state for that replay segment.
- Aggregate diagnostics remain backward-compatible:
  - `valid_mask_violation_count = mixed_stencil + hard_invalid`
  - split counts are reported separately for mixed and hard-invalid cases
- `retry_then_stop` is the default solver behavior so hard-invalid field samples are stopped before they can affect deposition or coating summaries.

## Boundary Event Direction

`max_wall_hits_per_step` is a legacy diagnostic guard. It is not an accuracy strategy.

- Production acceptance requires `boundary_event_contract_passed == 1`, which means `numerical_boundary_stopped_count == 0`, `unresolved_crossing_count == 0`, `max_hits_reached_count == 0`, and `nearest_projection_fallback_count == 0`.
- `nearest_projection_fallback_count > 0` is diagnostic debt. It should not be treated as a production success path.
- Boundary hits should be resolved from provider-backed primitive events plus a physical-time root solve on the particle trajectory.
- The provider tells the solver which primitive was hit; the solver computes when the trajectory hits it.
- `stuck` is a physical wall-model result, not a numerical fallback.
- Persistent wall contact must be represented as an explicit contact state, not as repeated reflection or max-hit retry.
- ContactState is part of the solver contract for 2D and 3D. It is not controlled by the obsolete `solver.contact_tangent_motion` option.
- Contact particles advance tangentially on the same boundary primitive, release only after a clean inside probe, and stop explicitly at endpoints, edges, or corners.
- Endpoint, edge, and corner ambiguity must be explicit in boundary-event diagnostics; it must not be hidden by projection fallback.
- The active implementation guide for current performance work is
  `plans/boundary_performance_plan.md`.

## Process-Step Scope

Process steps are retained as lightweight optional time labels for existing examples and source-event binding. They are not the core accuracy model for this package.

- Step rows must have finite times and positive duration (`end_s > start_s`).
- Zero-duration marker steps are intentionally unsupported.
- Gap-free, full-coverage process definitions are not required by the solver core.
- Missing process-step coverage falls back to the normal run step, keeping the primary contract centered on field sampling and boundary handling.

## Current Geometry Truth Contract

- 2D truth source is `boundary_loops_2d` when available, otherwise SDF fallback.
- 3D truth source is a validated closed triangle surface when available, otherwise SDF fallback.
- `BoundaryService` is the current internal contract for `inside`, `inside_strict`, `segment_hit`, `polyline_hit`, and diagnostic projection.
- `BoundaryHit` carries primitive identity, primitive kind, endpoint/corner ambiguity, and local signed-distance evaluation for provider-backed boundary events.
- 2D loop truth uses even-odd parity across disjoint loops, so nested-hole cases covered by regression tests are supported.
- 2D boundary-edge inputs are required to form disjoint degree-2 loops; shared-vertex branching or dangling topologies are rejected early instead of being silently reconstructed.
- 3D provider boundary checks sample representative triangle face, edge, and vertex neighborhoods before solving.
