# Generic COMSOL Particle Export

External tooling for reading a COMSOL `.mph` file and exporting the data needed
to build or validate a particle-tracing solver case.

This directory is intentionally outside `particle_tracer_unified`. The solver
package must not import COMSOL libraries or require COMSOL at runtime.

## Scope

The tool automates the parts that can be safely automated:

- load an `.mph` file through the COMSOL Java API
- write model, component, material, selection, and expression inventories
- optionally export a mesh text file
- optionally sample configured field expressions on a regular grid
- validate the raw export shape before a solver case is packed

The tool does not guess physical meaning. Wall laws, source surfaces, outlet
behavior, and COMSOL Particle Tracing feature equivalence must be reviewed from
the exported inventories and case-specific mapping files.

## Run Inventory And Field Export

```powershell
.\external\comsol_particle_export\run_export.ps1 `
  -ComsolExe "C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64\comsolbatch.exe" `
  -Mph "data\your_model.mph" `
  -Config "external\comsol_particle_export\config\export_case.example.json" `
  -OutDir "_external_exports\your_case"
```

The Java step writes files such as:

- `model_inventory.json`
- `material_inventory.json`
- `selection_inventory.json`
- `physics_feature_inventory.json`
- `particle_release_inventory.json`
- `expression_inventory.json`
- `export_manifest.json`
- `mesh.mphtxt` when `export_mesh` is enabled
- `field_samples.csv` when `export_fields` is enabled

## Validate Raw Export

```powershell
py -3 external\comsol_particle_export\validate_export.py `
  --raw-export-dir "_external_exports\your_case" `
  --config "external\comsol_particle_export\config\export_case.example.json" `
  --summary-out "_external_exports\your_case\raw_export_validation.json"
```

Validation checks file and data contracts plus the fields required by enabled
force models in `force_models`. It does not decide whether a boundary is wafer,
wall, outlet, inactive geometry, or source. Those choices belong in the
solver-case packing step after review.

## Compare COMSOL Particle Results

When COMSOL Particle Tracing results are exported as CSV, compare them with a
solver output directory using:

```powershell
py -3 external\comsol_particle_export\compare_particle_results.py `
  --solver-output-dir "_out_your_solver_run" `
  --comsol-particle-csv "_external_exports\your_case\comsol_particle_results.csv" `
  --raw-export-dir "_external_exports\your_case" `
  --solver-particles-csv "examples\your_case\particles.csv" `
  --comsol-release-csv "_external_exports\your_case\comsol_release_particles.csv" `
  --boundary-map-csv "_external_exports\your_case\boundary_id_map.csv" `
  --out-dir "_external_exports\your_case\comparison"
```

The comparison writes:

- `comparison_summary.json`
- `comparison_by_state.csv`
- `comparison_by_boundary.csv`
- `matched_particle_errors.csv`
- `force_model_alignment.json`
- `release_alignment.json`

The tool compares final state, first-hit boundary, hit time, hit position,
final position, final velocity, and charge when those columns are present. It
also reports whether COMSOL force features found in
`physics_feature_inventory.json` have corresponding enabled solver force
models and whether required exported fields are present. It does not tune
solver settings or infer wall physics.

`particle_release_inventory.json` is a review artifact for COMSOL release,
inlet, grid-release, initial-velocity, and particle-property features. It
records feature tags, labels, selection entities, available property names, and
common time/grid settings such as `tlist`, `releaseTime`, `Nx`, `Ny`, initial
position, velocity, diameter, density, mass, and charge when the COMSOL Java API
exposes them. Time-varying grid release should be reviewed here before packing
or regenerating `particles.csv`.

## Config Shape

The example config keeps the schema deliberately small:

- `case_name`
- `spatial_dim`
- `dataset`
- `mesh_tag`
- `axis_names`
- `axis_0_min`, `axis_0_max`, `axis_0_count`
- `axis_1_min`, `axis_1_max`, `axis_1_count`
- optional `axis_2_*` for 3D
- `required`
- `force_models`
- expression candidate lists such as `ux`, `uy`, `rho_g`, `T`, `E_x`

For COMSOL Particle Tracing reproducibility, keep force inputs explicit:

- thermophoresis needs `T`, `rho_g`, and `mu`; gradients are computed by the
  solver from `T`
- dielectrophoresis needs `E_x`, `E_y`, and `E_z` for 3D; `grad(|E|^2)` is
  computed by the solver
- Saffman lift needs velocity, `rho_g`, and `mu`; vorticity is computed by the
  solver
- gravity needs no field unless buoyancy is enabled, in which case `rho_g` is
  required

Keep expression candidates in the external config instead of adding model-name
knowledge to the solver.
