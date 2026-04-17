# ICP CF4/O2 COMSOL Export

External bridge for turning the COMSOL model
`icp_rf_bias_cf4_o2_si_etching (2).mph` into solver-ready geometry,
field, and particle inputs.

This directory is intentionally outside `particle_tracer_unified`.
The solver package must not import COMSOL libraries or require COMSOL
at runtime. COMSOL is used only to create export artifacts.

## Flow

1. Run the Java exporter on a machine with COMSOL installed.
2. Convert the exported samples into a strict field bundle.
3. Pack the mesh, field bundle, and generated particles into a solver case.

```powershell
.\external\comsol_icp_export\run_export.ps1 `
  -ComsolExe "C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64\comsolbatch.exe" `
  -Mph "data\icp_rf_bias_cf4_o2_si_etching (2).mph" `
  -OutDir "_external_exports\icp_cf4_o2_v20"

py -3 external\comsol_icp_export\pack_solver_case.py `
  --raw-export-dir "_external_exports\icp_cf4_o2_v20" `
  --out-dir "examples\icp_rf_bias_cf4_o2_si_etching_2d" `
  --particle-count 1000
```

After reviewing `generated/comsol_boundary_entity_mapping.csv`, confirmed wall
laws can be applied with an optional override CSV:

```powershell
py -3 external\comsol_icp_export\pack_solver_case.py `
  --raw-export-dir "_external_exports\icp_cf4_o2_v20" `
  --out-dir "examples\icp_rf_bias_cf4_o2_si_etching_2d" `
  --particle-count 1000 `
  --wall-overrides-csv "path\to\wall_catalog_overrides.csv"
```

`wall_catalog_overrides.csv` is intentionally part-ID based. The packer does
not infer wall physics from material names. Use the same columns as
`part_walls.csv` and include only COMSOL part IDs that have been reviewed.
For review, the packer also writes `generated/wall_catalog_review.csv`, which
joins current wall laws, COMSOL edge IDs, adjacent domain IDs, geometry bounds,
and exported material names when available.

The packer writes a short starter solver window:

```text
dt = 2e-8 s
t_end = 2e-6 s
```

Treat `dt`, `t_end`, particle count, and output cadence as case physics.
Do not reuse this starter window as a production benchmark without checking
particle response time, expected flight distance, RF/field time scales, and
wall-event statistics for the exported model.

## Raw Export Contract

The COMSOL step writes:

- `mesh.mphtxt`
- `field_samples.csv`
- `expression_inventory.json`
- `export_manifest.json`
- `material_inventory.json`

`field_samples.csv` must contain a complete tensor grid with columns:

- `r`, `z`
- `valid_mask`
- `ux`, `uy`, `mu`
- `E_x`, `E_y`

Optional diagnostic columns are preserved in the `.npz` bundle when present:
`T`, `p`, `rho_g`, `phi`, `ne`, `Te`, `ion_flux_*`, `etch_rate`.

The packer keeps electric field as `E_x` and `E_y`. The solver computes
electric acceleration from the current particle state as `(q(t)/m)E`, so the
field bundle must not precompute acceleration from a fixed reference charge/mass.
The expression map, grid shape, and source model information are recorded in
generated manifests.

The ICP model geometry is stored in COMSOL model coordinates of centimetres.
The raw export keeps those coordinates unchanged, and the packer converts
geometry, field axes, and particle positions to SI metres with:

```text
geometry_scale_m_per_model_unit = 0.01
```

Exported field quantities such as velocity, viscosity, electric field, gas
density, and temperature are kept in their COMSOL-evaluated physical units.

## Solver Case Contract

The generated case uses the existing `precomputed_npz` provider schema:

- `generated/comsol_geometry_2d.npz`
- `generated/comsol_field_2d.npz`
- `generated/comsol_boundary_entity_mapping.csv`
- `generated/comsol_domain_entity_mapping.csv`
- `materials.csv`
- `part_walls.csv`
- `particles.csv`
- `run_config.yaml`

`provider_contract.boundary_field_support: strict` and
`input_contract.initial_particle_field_support: strict` are enabled by default.

The entity mapping CSVs keep the COMSOL mesh entity IDs used by the solver
boundary catalog. COMSOL `mesh.mphtxt` contains geometric entity IDs but not
material selection names. When `material_inventory.json` is available, the
packer copies its material-to-domain selections into the boundary/domain
mapping CSVs. When it is not available, `comsol_material_name` remains
`not_exported_from_mphtxt` instead of guessing.
The packer checks whether COMSOL material selection entity IDs need a simple
`0`, `+1`, or `-1` offset to match exported mesh domain IDs, and records the
chosen offset in `generated/icp_export_case_manifest.json`.

Wall behavior is applied from `part_walls.csv`. For production cases, review
the generated entity mapping and use `--wall-overrides-csv` for confirmed
part-specific physics. Unspecified parts keep the conservative generated
defaults.

## Scope Notes

- This external bridge currently targets the 2D axisymmetric r-z RF bias
  `20 V` ICP case.
- The solver case uses fluid velocity, gas properties, and electric field.
- COMSOL expression discovery is fail-fast: if required fields are not found,
  no field export is produced.
- The solver hot path does not consume COMSOL-only diagnostics; they are kept
  for review and visualization.
