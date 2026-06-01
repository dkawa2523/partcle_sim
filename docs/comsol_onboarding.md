# COMSOL Onboarding Runbook

This is the short path for bringing a COMSOL particle case into the solver
without adding hidden repair logic.

## Solver Inputs

The solver consumes exported, explicit artifacts:

- `run_config.yaml`
- COMSOL case manifest YAML
- CSV tables for release particles, particle results, boundary maps, and wall laws
- NPZ provider bundles for geometry and fields

The solver must not consume `.mph` files or call COMSOL APIs inside
`particle_tracer_unified`. Export scripts and COMSOL API code belong in
`external/` or `tools/`; runtime loading belongs in `particle_tracer_unified/io/`.

## Export Checklist

A faithful comparison case needs these exported inputs before solver debugging:

- Release table: raw COMSOL release coordinates, velocities, release times, and
  source identifiers.
- Particle result table: COMSOL trajectory/result data for comparison.
- Boundary map: COMSOL geometric entities mapped to solver boundary part IDs.
- Wall law map: wall behavior for every boundary part used by the boundary map.
- Field bundle or field mappings: velocity, electric field, and any other mapped
  quantities needed by the enabled forces.
- Coordinate system and scale: `cartesian_xy`, `axisymmetric_rz`, or
  `cartesian_xyz`, plus `coordinates.coordinate_scale_m_per_model_unit`.
- Force inventory: enabled COMSOL forces, solver force names, drag law, and
  physical quantity labels.

Use `examples/comsol_faithful_2d/` as the manifest-first template. The template
is runnable only after its referenced exported CSV/NPZ files exist.

## Choose The Mode

Use `comsol_faithful` when the goal is machine parity with COMSOL. This mode
preserves COMSOL release coordinates, rejects source preprocessing and boundary
snap/repair, requires coordinate scale, wall laws, field mappings, and force
inventory, and keeps strict field support.

Use `surface_release_production` when the goal is a production chamber run from
operator-provided surface-origin particles. This mode can use explicit
`source.preprocess.boundary_release`, capture tolerance, projection, and a small
inward offset, but it is not a faithful replay of COMSOL release coordinates.

Optional example: a VIGUS-scale case may use either path depending on the goal;
do not encode VIGUS-specific wall IDs or part IDs in solver core.

## Comparison Order

Work from the earliest artifact outward:

1. Import: manifest, coordinates, field bundle, boundary map, wall law map.
2. Preprocess: release particles and any allowed source preprocessing.
3. First-step: deterministic force and first-step comparison.
4. Wall events: boundary hit timing, part IDs, and wall-law outcomes.
5. Ensemble: final distributions, event fractions, and run summaries.

Useful commands:

```powershell
particle-tracer-build-comsol-case --help
py -3 run_from_yaml.py <case>/run_config.yaml --check-input --output-dir <out_check>
py -3 run_from_yaml.py <case>/run_config.yaml --output-dir <out_run>
particle-tracer-field-compare --help
particle-tracer-acceleration-compare --help
particle-tracer-boundary-compare --help
particle-tracer-trajectory-compare --help
```

For first-step comparison:

```powershell
py -3 -m particle_tracer_unified.compare.first_step_compare --config <case>/run_config.yaml --output-dir <out_first_step>
```

## Failure Triage

If field support fails, fix the export or provider bundle first. Do not expand
valid masks, add ghost cells, clamp coordinates, or extrapolate in the solver to
hide an export gap.

If release import fails, fix the release table, coordinate scale, coordinate
system, or manifest path. In `comsol_faithful`, do not snap or repair release
coordinates; preserve them or fail clearly.

If wall-law or boundary comparison fails, fix the boundary map and wall-law
coverage before changing wall physics. Every mapped boundary part must have an
explicit wall law.

Solver-side repair for bad COMSOL export makes parity impossible to interpret:
it changes the artifact being compared, masks export mistakes, and can make a
local endpoint count look better while degrading the global time evolution.
