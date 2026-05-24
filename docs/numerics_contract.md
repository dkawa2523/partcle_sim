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

Internal field-sampling helpers may bundle multiple requested quantities at a
stage point, but they must call the same provider/backend sampling paths and
preserve the provider-authored status. A bundled sample is a cache/organization
device, not permission to extrapolate, repair, or reinterpret field support.

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

Surface release preprocessing is the exception for wall-origin particles. When
`source.preprocess.boundary_release` is enabled, particles that start on or very
near a boundary are classified against the explicit boundary primitives and are
offset a small distance into the simulated domain before the input preflight.
This keeps part-origin flake/resuspension cases from treating a valid wall
release as an invalid field point. `hard_invalid` remains terminal; COMSOL
faithful mode keeps strict clean support.

| Setting | Role |
| --- | --- |
| `source.preprocess.boundary_release` | Normalize boundary source particles into the domain. |
| `input_contract.initial_particle_field_support` | Schema key for the initial field-support preflight after preprocessing. Use `warn` for boundary release studies. |
| `solver.valid_mask_policy` | Runtime response to field support loss. Use `retry_then_stop` for production trajectories. |
| `field_support.mixed_stencil_policy` | COMSOL faithful export gate; keep `error` for strict parity. |

## Calibration Inputs

Treat these settings as case calibration, not numerical rescue knobs:

- `solver.epsilon_offset_m` and `solver.on_boundary_tol_m` control boundary
  release/contact tolerance and should be scaled to geometry resolution.
- Wall law probabilities, restitution, and diffuse/stick choices should come
  from material or chamber assumptions, not from trajectory cleanup.
- `solver.drag_model` selects the drag law. Scalar gas fallbacks are used only
  where the existing model already permits them; they do not fill field gaps.

## Boundaries

Wall events should come from provider-backed boundary primitives plus a physical
hit-time solve on the particle trajectory. Wall reflection uses the hit-time
state `(x_hit, v_hit)`, not the segment endpoint.

Boundary broad-phase pruning is allowed only as a conservative candidate filter
in front of the existing exact hit-time solve. Candidate misses must remain zero
in debug diagnostics, and COMSOL faithful mode keeps broad-phase pruning disabled
by default until boundary comparison parity has been demonstrated.

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

Dynamic charge currently supports 2D regular-rectilinear field inputs or scalar
plasma-background inputs. 3D and triangle-mesh charge updates are outside the
current runtime update path. COMSOL flux vectors are not consumed directly by
the charge model.

Near-wall drag or lift corrections are not automatically applied. If sheath,
ion-drag, near-wall lift, or other chamber-specific effects are required, supply
them through fields, existing force inputs, or explicit source/wall data rather
than expecting the solver to infer them.

## Surface-Origin Scope

| Solver handles | User supplies |
| --- | --- |
| Trajectories of particles already released into supplied fields. | Whether a part fractures, flakes, or sheds deposits. |
| Boundary release normalization from explicit boundary primitives. | Release population, particle sizes, initial speeds, and release timing. |
| Stokes, Schiller-Naumann, Cunningham, or Epstein drag. | Gas state and rarefaction-relevant inputs such as `rho_g`, `T`, and gas molecular mass. |
| Optional Brownian/stochastic motion, charge evolution, `qE/m`, thermophoresis, DEP, lift, pressure-gradient, and virtual-mass terms. | Flow, plasma, electric field, gradients/derived fields, wall law, and material data. |
| Wall hit-time and configured wall laws using particle-center geometry. | Sheath physics, ion-drag effects, near-wall corrections, or detachment probabilities unless represented by fields, forces, or source inputs. |

## Process Steps

`process_steps.csv` is a lightweight time-label overlay. Rows must have finite
times and positive duration. Missing coverage falls back to the normal `run`
step.
