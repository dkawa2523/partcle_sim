# Numerics Notes

This file is a short implementation guide, not a second specification. Keep the
solver behavior discoverable from the code and tests.

## Model

Particles advance in supplied geometry and provider-backed fields:

```text
dx/dt = v
dv/dt = (u(x,t) - v) / tau_eff + a_body + (q(t) / m) E(x,t)
```

Drag is selected with `solver.drag_model`:

- `stokes`: linear Stokes relaxation
- `schiller_naumann`: finite-Re drag correction from local slip speed
- `epstein`: low-pressure free-molecular relaxation using field `rho_g` and
  `T` when present, otherwise scalar `gas.*` fallbacks

Electric force is always evaluated in the solver as `(q(t)/m)E`. Field bundles
should provide electric fields, not fixed-charge acceleration.

## Integrators

Public `solver.integrator` values:

- `drag_relaxation`
- `etd`
- `etd2`

The implementation details live in
`particle_tracer_unified/core/integrator_registry.py` and the free-flight
kernels. Partial replay must use the same segment logic over a shorter `dt` so
collision replay and normal free flight stay consistent.

## Field Support

Provider support is authoritative. Runtime assembly does not fill missing field
values, clamp outside-axis points, or silently expand valid masks.

Valid-mask states:

- `clean`: point and interpolation stencil are valid
- `mixed_stencil`: point is valid but interpolation touches invalid nodes
- `hard_invalid`: point is outside valid field support

`solver.valid_mask_policy`:

- `retry_then_stop`: stop hard-invalid trial segments at the best valid prefix
- `strict_clean`: also treats mixed stencils as terminal
- `diagnostic`: record diagnostics only

Boundary-adjacent field gaps should be fixed in exported inputs or providers,
not hidden by solver rescue logic.

## Boundaries

Wall events should come from provider-backed boundary primitives plus a physical
hit-time solve on the particle trajectory. Wall reflection uses the hit-time
state `(x_hit, v_hit)`, not the segment endpoint.

`max_wall_hits_per_step` is a diagnostic guard. Production runs should have:

- `numerical_boundary_stopped_count == 0`
- `unresolved_crossing_count == 0`
- `max_hits_reached_count == 0`
- `nearest_projection_fallback_count == 0`

Persistent contact is represented explicitly as contact state. It should not be
modeled as repeated reflection.

## Geometry Truth

- 2D uses boundary loops when available, otherwise SDF fallback.
- 3D uses validated closed triangle surfaces when available, otherwise SDF
  fallback.
- `BoundaryService` is the shared internal entry point for inside checks,
  segment hits, polyline hits, and diagnostic projection.

## Optional Physics

Stochastic motion and charge evolution are opt-in model behavior. They must not
act as corrections for missing fields or boundary failures.

2D dynamic charge currently supports field-backed or scalar plasma-background
inputs. COMSOL flux vectors are not consumed directly by the charge model.

## Process Steps

`process_steps.csv` is a lightweight time-label overlay. Rows must have finite
times and positive duration. Missing coverage falls back to the normal `run`
step.
