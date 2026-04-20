# COMSOL Case Handoff For Another Copilot

Status: operational handoff guide for a different environment and a different
COMSOL `.mph` file.

Use this file when another Copilot needs to import a new COMSOL model, understand
its particle-tracing setup, create a solver-ready case, run it, compare it with
COMSOL, and evaluate whether differences are caused by export data, particle
release, force models, wall laws, or numerical settings.

## Core Rule

The solver package must stay independent from COMSOL. COMSOL API code and raw
COMSOL outputs belong under `external/` and `_external_exports/`. Solver-ready
inputs are plain `run_config.yaml`, CSV, and `.npz` files.

Do not add solver-side rescue logic to hide bad export data. If a case fails,
classify the cause first:

- geometry or coordinate scale
- field expression, unit, sign, or support
- particle release setup
- force model mismatch
- wall law or boundary mapping
- numerical time step or output cadence

## Required Inputs From The User

Ask for these before judging results:

- COMSOL `.mph` path and COMSOL version
- target study, dataset, parameter point, and time/solution index
- coordinate system: planar 2D, axisymmetric r-z, or true 3D
- model coordinate unit and SI scale
- particle material, diameter, density, mass, charge or charge model
- source/release feature identity and whether release is time varying
- intended COMSOL Particle Tracing force nodes
- wall interactions for wafer, sidewall, chamber wall, outlet, symmetry axis,
  inactive/internal parts, and support-only regions
- COMSOL particle result CSV for comparison, if reproducibility is required
- COMSOL release particle CSV or table, especially for grid/time-dependent
  release

Stop and ask when source surface, wall law, coordinate scale, particle charge,
or COMSOL release timing is ambiguous. These are physical settings, not code
cleanup items.

## Directory Convention

Use predictable paths so another environment can reproduce the same workflow:

- source `.mph`: `data/<case_name>.mph`
- raw COMSOL export: `_external_exports/<case_name>/`
- solver-ready case: `examples/<case_name>/`
- solver outputs: `_out_<case_name>_<run_label>/`
- report figures only when needed: `report_assets/<case_name>/`

Do not commit large local output folders unless the user explicitly asks for a
report bundle.

## Phase 1: Raw COMSOL Export

Start with the generic external exporter:

```powershell
.\external\comsol_particle_export\run_export.ps1 `
  -ComsolExe "C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64\comsolbatch.exe" `
  -Mph "data\<case_name>.mph" `
  -Config "external\comsol_particle_export\config\export_case.example.json" `
  -OutDir "_external_exports\<case_name>"
```

The export should produce:

- `model_inventory.json`
- `material_inventory.json`
- `selection_inventory.json`
- `physics_feature_inventory.json`
- `particle_release_inventory.json`
- `expression_inventory.json`
- `export_manifest.json`
- `mesh.mphtxt`, when mesh export is enabled
- `field_samples.csv`, when field export is enabled

If field expressions are unknown, run inventory first or use a broad expression
candidate list. Do not hard-code model-specific variable names in the solver.

## Phase 2: Export Config Checklist

Edit `external/comsol_particle_export/config/export_case.example.json` or copy it
to a case-specific config. Make these explicit:

- `case_name`
- `spatial_dim`
- `dataset`
- `mesh_tag`
- `axis_names`
- axis bounds and counts
- `coordinate_model_unit`
- `coordinate_scale_m_per_model_unit`
- `required`
- `force_models`
- expression candidates for fields

Required fields for common modes:

- base trajectory: `ux`, `uy`, optionally `uz`, `mu`, `valid_mask`
- electric force: `E_x`, `E_y`, optionally `E_z`
- low-pressure Epstein drag: `rho_g`, `T`, `mu`
- Brownian/Langevin: `T`, gas properties
- thermophoresis: `T`, `rho_g`, `mu`
- dielectrophoresis: `E_x`, `E_y`, optionally `E_z`
- Saffman lift: velocity, `rho_g`, `mu`
- gravity with buoyancy: `rho_g`

Gradients should normally be computed by the solver from exported fields:

- `grad(T)` for thermophoresis
- `grad(|E|^2)` for dielectrophoresis
- velocity curl/vorticity for Saffman lift

Only export explicit COMSOL gradients when reproducing a custom COMSOL force
expression that cannot be reconstructed from these fields.

## Phase 3: Validate Raw Export

Run:

```powershell
py -3 external\comsol_particle_export\validate_export.py `
  --raw-export-dir "_external_exports\<case_name>" `
  --config "external\comsol_particle_export\config\<case_config>.json" `
  --summary-out "_external_exports\<case_name>\raw_export_validation.json"
```

Review:

- `field_samples.row_count` matches a complete tensor grid
- required field quantities are finite on valid support
- selected optional fields have plausible min/max/variation
- `force_requirements` passes for enabled force models
- `particle_release_inventory.feature_count` is not zero when COMSOL has
  release features
- `particle_release_inventory.time_dependent_feature_count` is nonzero when
  COMSOL uses time-dependent grid release

If validation fails, fix the export/config before building a solver case.

## Phase 4: Review COMSOL Particle Setup

Open and inspect:

- `_external_exports/<case_name>/physics_feature_inventory.json`
- `_external_exports/<case_name>/particle_release_inventory.json`
- `_external_exports/<case_name>/selection_inventory.json`
- `_external_exports/<case_name>/material_inventory.json`

For particle release, record:

- release feature tag and label
- selection entity IDs
- release kind: grid, boundary, domain, inlet, point, initial coordinates
- release times: `tlist`, `releaseTime`, `period`, or similar settings
- grid dimensions such as `Nx`, `Ny`, `Nz`, `Nr`
- source coordinates or boundary selection
- initial speed/vector/direction/normal convention
- particle count or flux/probability weighting
- diameter, density, mass, charge, material, and distribution

For time-varying grid release, do not assume uniform release. Prefer exporting a
COMSOL release particle table with `particle_id`, release time, initial
position, initial velocity, source boundary/entity, and any release weight.
This table can be compared with solver `particles.csv` by the external compare
tool.

## Phase 5: Build Solver-Ready Case

A solver-ready case must contain:

- `run_config.yaml`
- `materials.csv`
- `part_walls.csv`
- `particles.csv`
- `generated/comsol_geometry_*.npz`
- `generated/comsol_field_*.npz`

For the existing ICP bridge, use:

```powershell
py -3 -m external.comsol_icp_export.comsol_icp_export.pack_solver_case `
  --raw-export-dir "_external_exports\<case_name>" `
  --out-dir "examples\<case_name>" `
  --particle-count 1000
```

For a different COMSOL family, only reuse this packer if the geometry, units,
and source assumptions match. Otherwise create or adapt an external packer under
`external/<case_exporter>/`; do not add COMSOL assumptions inside
`particle_tracer_unified`.

Standard production safety settings:

```yaml
provider_contract:
  boundary_field_support: strict
input_contract:
  initial_particle_field_support: strict
output:
  artifact_mode: minimal
```

## Phase 6: Preflight Solver Check

Before a production run:

```powershell
py -3 run_from_yaml.py examples\<case_name>\run_config.yaml `
  --prepare-only `
  --output-dir "_out_<case_name>_prepare"
```

Inspect:

- `prepared_runtime_summary.json`
- source geometry summary, if written
- provider/input contract reports when available
- material and wall mapping
- force catalog/runtime summary after a short run

Do not proceed if initial particles are outside clean field support, source
parts are wrong, or boundary parts are misidentified.

## Phase 7: Smoke Run, Then Production

Use a small smoke run first:

- 100 to 1000 particles
- short `t_end`
- minimal artifacts
- strict provider/input contracts

Then run 10k or production only after smoke results pass physical checks.

Important physical checks:

- particle release direction is correct
- field scale and sign produce plausible acceleration
- drag uses field `rho_g` and `T` when the case is low pressure
- charge sign and magnitude are plausible
- wall outcomes match the intended wall law
- `invalid_mask_stopped`, `numerical_boundary_stopped`, and stuck-at-numerics
  counts are not hiding export/boundary mistakes

## Phase 8: Compare With COMSOL Particle Tracing

When COMSOL particle result CSV exists:

```powershell
py -3 external\comsol_particle_export\compare_particle_results.py `
  --solver-output-dir "_out_<case_name>_<run_label>" `
  --comsol-particle-csv "_external_exports\<case_name>\comsol_particle_results.csv" `
  --raw-export-dir "_external_exports\<case_name>" `
  --solver-particles-csv "examples\<case_name>\particles.csv" `
  --comsol-release-csv "_external_exports\<case_name>\comsol_release_particles.csv" `
  --boundary-map-csv "_external_exports\<case_name>\boundary_id_map.csv" `
  --out-dir "_external_exports\<case_name>\comparison_<run_label>"
```

Expected comparison outputs:

- `comparison_summary.json`
- `comparison_by_state.csv`
- `comparison_by_boundary.csv`
- `matched_particle_errors.csv`
- `force_model_alignment.json`
- `release_alignment.json`

Review order:

1. `release_alignment.json`: release time, position, velocity, source entity.
2. `force_model_alignment.json`: COMSOL force nodes vs enabled solver forces.
3. `comparison_by_state.csv`: state/outcome mismatch.
4. `comparison_by_boundary.csv`: hit boundary/entity mismatch.
5. `matched_particle_errors.csv`: hit time, position, velocity, charge error.

If release alignment fails, fix `particles.csv` or the case packer before
changing force models. If force alignment fails, fix solver force settings or
field export before tuning time steps.

## Evaluation Criteria

For a COMSOL reproduction task, consider the case usable only when:

- geometry scale and coordinate system are documented
- material and boundary entity mapping is explicit
- release settings are either reproduced or intentionally replaced
- all COMSOL force nodes have solver counterparts or documented exclusions
- required fields are finite on valid support
- wall laws are mapped by physical part/material, not by visual guess
- smoke run has no provider/input contract failure
- comparison metrics identify remaining differences without solver rescue logic

For a production solver task that does not need exact COMSOL reproduction, still
preserve the same inventories so later comparisons remain possible.

## What Not To Do

- Do not infer time-dependent grid release from final particle plots.
- Do not replace missing COMSOL release data with uniform particles unless the
  user accepts that approximation.
- Do not silently reinterpret COMSOL boundary IDs as solver part IDs without a
  boundary map.
- Do not use lower deposition or reflection counts as proof of correctness.
- Do not add case-specific hacks to collision handling or valid-mask retry.
- Do not put COMSOL Java or COMSOL variable-name assumptions into the solver
  package.

## Handoff Summary Template

Copy this block into a case note before handing off:

```text
Case name:
COMSOL mph:
COMSOL version:
Study/dataset/solution:
Coordinate system and scale:
Raw export dir:
Solver case dir:
Solver output dir:

Geometry/boundary notes:
Material mapping:
Wall laws:

Particle model:
Release feature(s):
Release timing:
Release grid/selection:
Initial velocity:

Enabled solver forces:
COMSOL force features:
Fields exported:
Fields missing:

Smoke result:
Production result:
Comparison outputs:
Known mismatches:
Next action:
```
