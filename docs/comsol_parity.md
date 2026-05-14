# COMSOL Faithful Parity

`comsol_faithful` mode is a manifest-first comparison path. It is meant for
machine comparison of release particles, coordinate scale, field samples, force
inventory, and boundary events. It is not a visual "looks similar" mode.

## Required Inputs

The run config must set:

```yaml
mode: comsol_faithful
comsol:
  manifest: comsol_case_manifest.yaml
```

The manifest is the single source of COMSOL comparison metadata. In strict mode
it must provide:

- model/study/dataset/solution metadata
- `coordinates.coordinate_system`
- `coordinates.coordinate_scale_m_per_model_unit`
- field mappings with `physical_quantity`
- `particles.release_table`
- `boundaries.map_file`
- `boundaries.wall_law_file`
- explicit enabled force inventory, including drag law

The solver core never reads `.mph` files. COMSOL API/export work belongs under
`external/` or `tools/`.

Runtime integration is centralized in:

```text
particle_tracer_unified/io/comsol.py
```

The example under `examples/comsol_faithful_2d` is a template. It becomes
runnable only after the referenced exported CSV/NPZ files are generated or
copied into the case.

## Faithful Gates

Faithful mode rejects:

- source preprocessing
- missing coordinate scale
- implicit release particle generation
- missing/unknown wall laws
- missing force inventory or missing drag law
- field ghost cells
- mixed stencil policy other than `error`

The default field policy is strict clean support. `mixed_stencil` and
`hard_invalid` samples are diagnostics failures, not soft warnings.
Normal production runs can use `source.preprocess.boundary_release` for
wall-origin particles, but that preprocessing remains outside faithful mode so
COMSOL release coordinates stay machine-comparable.

## Field Backend Mode

Use `solver.field_backend_mode` to make the field backend explicit:

```yaml
solver:
  field_backend_mode: auto        # auto | regular_grid | triangle_mesh
```

`regular_grid` requires a rectilinear field bundle. `triangle_mesh` requires a
2D triangle-mesh field bundle. Extra forces that depend on field derivatives
or particle-path fluid acceleration use the shared precise substep evaluator so
they are sampled at the particle state inside each step. Triangle mesh drag can
still use the fast mesh kernel when no extra force or electric field is active.

## Comparison CLIs

Install entry points expose:

```powershell
particle-tracer-field-compare --help
particle-tracer-acceleration-compare --help
particle-tracer-trajectory-compare --help
particle-tracer-boundary-compare --help
```

They can also be run as modules:

```powershell
python -m particle_tracer_unified.compare.field_compare --help
python -m particle_tracer_unified.compare.acceleration_compare --help
python -m particle_tracer_unified.compare.trajectory_compare --help
python -m particle_tracer_unified.compare.boundary_compare --help
```

Expected diagnostics include `run_summary.json`, `field_validation_error.csv`,
`acceleration_error.csv`, `trajectory_error.csv`,
`boundary_hit_comparison.csv`, `collision_diagnostics.json`,
`force_contributions.csv`, and `provider_boundary_violations.csv` when the
corresponding validation path is executed.
