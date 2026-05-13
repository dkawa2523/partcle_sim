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
- optionally sample configured field expressions on a regular grid, including
  multiple `solnums` or requested `time_values`
- optionally sample configured field expressions on COMSOL mesh vertices for
  the solver's `triangle_mesh_2d` backend
- optionally run a configured COMSOL Data export against a result dataset, such
  as a Particle 2D dataset, to capture raw trajectory/result tables
- build solver-ready NPZ field bundles from `field_samples.csv` or
  `mesh_field_samples.csv`
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
- `study_inventory.json`
- `dataset_inventory.json`
- `physics_feature_inventory.json`
- `particle_release_inventory.json`
- `expression_inventory.json`
- `export_manifest.json`
- `data_export_report.json` when `export_data_table` is enabled
- `mesh.mphtxt` when `export_mesh` is enabled
- `field_samples.csv` when `export_fields` is enabled
- `mesh_field_samples.csv` when `export_mesh_field_samples` is enabled
- the configured Data export CSV when `export_data_table` is enabled and
  COMSOL can export the requested dataset/expression combination

For inventory-only onboarding, use an empty `required` list and disable field
export. The repository includes
`external/comsol_particle_export/config/micromixer_inventory.json` as a clean
starting point for `data/micromixer_particle_tracing.mph`.

## Validate Raw Export

```powershell
py -3 external\comsol_particle_export\validate_export.py `
  --raw-export-dir "_external_exports\your_case" `
  --config "external\comsol_particle_export\config\export_case.example.json" `
  --summary-out "_external_exports\your_case\raw_export_validation.json"
```

Validation checks file and data contracts plus the fields required by enabled
force models in `force_models`. It does not decide whether a boundary is a
wall, outlet, inactive geometry, or source. Those choices belong in the
solver-case packing step after review.

When `field_samples.csv` contains `time_s`/`time`/`t` or `solnum`, validation
requires every time/solution context to contain a complete tensor grid.
Optional `comsol_release_particles.csv` and `comsol_particle_results.csv` are
validated when present; set `require_release_table` or
`require_particle_results` in the config to make missing tables fail.

## Build A Field Bundle

After exporting fields, convert the sample table to solver NPZ with:

```powershell
py -3 external\comsol_particle_export\build_field_bundle.py `
  --field-samples-csv "_external_exports\your_case\field_samples.csv" `
  --out-npz "_external_exports\your_case\comsol_field_nd.npz" `
  --axis-names x y z `
  --quantities ux uy uz mu
```

The builder writes quantity arrays as `(nt, *spatial_shape)` and stores a
time-independent `valid_mask` as the intersection of valid support over all
exported contexts.

For mesh-native 2D FEM field replay, enable `export_mesh_field_samples` in the
COMSOL export config and build a triangle mesh bundle:

```powershell
py -3 external\comsol_particle_export\build_mesh_field_bundle.py `
  --mesh-field-samples-csv "_external_exports\your_case\mesh_field_samples.csv" `
  --geometry-npz "examples\your_case\generated\comsol_geometry_2d.npz" `
  --out-npz "examples\your_case\generated\comsol_field_mesh_2d.npz" `
  --axis-names x y `
  --quantities ux uy mu rho_g p `
  --coordinate-scale-m-per-model-unit 0.001 `
  --coordinate-model-unit mm `
  --support-tolerance-m 2.5e-5
```

The mesh builder requires one sample per mesh vertex per time context and fails
on missing quantities, mismatched vertex coordinates, non-finite required
values, or unsupported non-2D mesh bundles. `support_tolerance_m` is a metric
distance tolerance used by `triangle_mesh_2d` for wall-near numerical drift.
For time-resolved mesh-only field extraction, set
`export_grid_field_samples=false` and `export_mesh_field_samples=true` so the
export does not produce a large regular-grid table that is not used for
faithful boundary-near replay.

## Probe A COMSOL Result Dataset

For particle trajectory datasets that are not field grids, enable the generic
Data export pass in the config:

```json
{
  "export_data_table": true,
  "data_export_dataset": "part1",
  "data_export_filename": "comsol_particle_data_probe.csv",
  "data_export_innerinput": "all",
  "data_export_expr": ["1"]
}
```

This uses COMSOL's own Data export feature and records success/failure plus the
requested dataset, expressions, time values, and solution numbers in
`data_export_report.json`. Keep the first pass simple, then promote a reviewed
raw CSV into canonical `comsol_particle_trajectory.csv` or
`comsol_particle_results.csv` only after its headers and row semantics are
understood.

## Compare COMSOL Particle Results

Release alignment can be checked before running the solver:

```powershell
py -3 external\comsol_particle_export\compare_release_tables.py `
  --solver-particles-csv "examples\your_case\particles.csv" `
  --comsol-release-csv "_external_exports\your_case\comsol_release_particles.csv" `
  --out-dir "_external_exports\your_case\release_alignment"
```

When COMSOL Particle Tracing results are exported as CSV, compare them with a
solver output directory using:

```powershell
py -3 external\comsol_particle_export\compare_particle_results.py `
  --solver-output-dir "_out_your_solver_run" `
  --comsol-particle-csv "_external_exports\your_case\comsol_particle_results.csv" `
  --raw-export-dir "_external_exports\your_case" `
  --solver-particles-csv "examples\your_case\particles.csv" `
  --comsol-release-csv "_external_exports\your_case\comsol_release_particles.csv" `
  --comsol-trajectory-csv "_external_exports\your_case\comsol_particle_trajectory.csv" `
  --boundary-map-csv "_external_exports\your_case\boundary_id_map.csv" `
  --field-npz "examples\your_case\generated\comsol_field_mesh_2d.npz" `
  --out-dir "_external_exports\your_case\comparison"
```

The comparison writes:

- `comparison_summary.json`
- `comparison_by_state.csv`
- `comparison_by_boundary.csv`
- `matched_particle_errors.csv`
- `force_model_alignment.json`
- `release_alignment.json`
- `trajectory_alignment.json`
- `matched_trajectory_errors.csv`
- `distribution_alignment.csv`
- `field_alignment.json`
- `field_alignment_by_source.csv`
- `field_alignment_by_time.csv`
- `boundary_role_alignment.json`
- `trend_alignment.json`
- `divergence_alignment.json`

The tool compares final state, first-hit boundary, hit time, hit position,
final position, final velocity, and charge when those columns are present. When
`--comsol-trajectory-csv` is supplied and the solver wrote `positions_2d.npy` or
`positions_3d.npy`, it also compares matched trajectory samples and per-time
centroid/RMS distribution summaries. When `--field-npz` is supplied, it replays
regular-grid or `triangle_mesh_2d` field values on the COMSOL trajectory and
reports support fraction plus velocity residuals. It
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

## Audit Existing Truth Exports

When COMSOL is not available on the machine, audit the existing export artifacts
before running new solver comparisons:

```powershell
py -3 external\comsol_particle_export\audit_truth_export.py `
  --case-name micromixer_particle_tracing `
  --field-raw-dir _external_exports\micromixer_particle_tracing_field_probe `
  --particle-raw-dir _external_exports\micromixer_particle_tracing_xy_velocity_probe `
  --solver-case-dir _external_exports\micromixer_particle_tracing_solver_case `
  --field-npz _external_exports\micromixer_particle_tracing_solver_case\generated\comsol_field_mesh_2d.npz `
  --regular-field-npz _external_exports\micromixer_particle_tracing_solver_case\generated\comsol_field_2d.npz `
  --out-dir _external_exports\micromixer_particle_tracing_truth_audit
```

The audit writes `micromixer_truth_manifest.yaml`,
`micromixer_truth_manifest.json`, and `micromixer_audit_summary.json`, plus
release, boundary-role, optional mesh-field replay diagnostics, and
`required_comsol_exports/reextract_request_summary.json`. It records which
files are truth inputs, which ones are diagnostic-only, which COMSOL exports
are still missing, and `parity_readiness.ready_for_exact_solver_comparison`.
Boundary diagnostics are split into particle stop-time/status truth and direct
wall-hit entity/normal truth. `fpt.st`/`fpt.fs` are promoted only to
`comsol_particle_status.csv`; they are not wall-hit entity/normal truth. If the
release table lacks row-level source or particle properties, the audit emits
one-expression probe configs so those variables can be discovered without one
unknown COMSOL expression failing the whole export.

The request bundle also writes `run_reextract_requests.ps1`, a thin wrapper
around `run_export.ps1`:

```powershell
.\_external_exports\micromixer_particle_tracing_truth_audit\required_comsol_exports\run_reextract_requests.ps1 `
  -ComsolExe "C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64\comsolbatch.exe" `
  -Mph "data\micromixer_particle_tracing.mph" `
  -OutRoot "_external_exports\micromixer_particle_tracing_reextract"
```

Keep successful probe outputs separate until their columns are promoted into a
canonical release or boundary-event table. Failed expression probes are useful
negative evidence and should not be mixed into solver input files. If the
canonical trajectory, release properties, particle status, time-resolved field,
and reviewed boundary evidence are already present, the summary can legitimately
contain zero runnable requests.

Promote reviewed re-extraction outputs with:

```powershell
py -3 external\comsol_particle_export\promote_reextract_outputs.py `
  --reextract-root "_external_exports\micromixer_particle_tracing_reextract" `
  --baseline-release-csv "_external_exports\micromixer_particle_tracing_xy_velocity_probe\comsol_release_particles.csv" `
  --particle-release-inventory-json "_external_exports\micromixer_particle_tracing_field_probe\particle_release_inventory.json" `
  --out-dir "_external_exports\micromixer_particle_tracing_reextract\canonical"
```

The promotion step writes `comsol_release_particles_canonical.csv`,
`comsol_particle_status.csv` when `fpt.st`/`fpt.fs` are available,
`comsol_wall_events.csv` only when a schema-valid wall-hit entity/normal table
is present, and `promotion_summary.json`. The micromixer source assignment is
derived from COMSOL release selections plus the reviewed boundary map; it is
recorded in the release promotion report.

For faithful micromixer debugging, keep these roles separate:

- exact COMSOL release: truth input
- inward-clean release: solver support diagnostic only
- mesh-native field bundle: primary field replay truth
- regular-grid field bundle and ghost cells: diagnostic only
- solver `_out_micromixer_*` directories: reference runs, not COMSOL truth

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
- optional `solnums` and `time_values` for multi-context field export
- optional `export_grid_field_samples=false` to skip regular-grid samples while
  still writing mesh-native field samples
- optional `export_data_table`, `data_export_dataset`,
  `data_export_filename`, `data_export_innerinput`, `data_export_solnums`,
  `data_export_time_values`, and `data_export_expr` for raw COMSOL Data export
- optional `require_release_table` and `require_particle_results`
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
