# ICP CF4/O2 COMSOL exporter

This directory contains only the model-specific COMSOL boundary: the Java
exporter, its profile data, and the PowerShell launcher needed on a COMSOL
installation.  Python packing is part of the generic application tool.

First review `config/icp_cf4_o2_v20.json`. The exporter deliberately has no
model defaults for provenance or physics. Set all of the following to the
actual solved model:

- `model_name`, `study`, `dataset`, `solution`, and `solution_number`;
- `mesh_tag`, the parameter name/value, and the non-empty
  `vacuum_domain_ids` occupied by particles;
- exactly one COMSOL expression and one requested unit for every exported
  semantic quantity.

The checked-in `vacuum_domain_ids` list is empty because a domain ID cannot be
inferred safely without reviewing the model. Export fails until it is filled.
There is no expression candidate search and no fallback to a speed magnitude,
another potential, or another solution.

Then export the model:

```powershell
.\external\comsol_icp_export\run_export.ps1 `
  -ComsolExe "C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64\comsolbatch.exe" `
  -Mph "data\icp_rf_bias_cf4_o2_si_etching (2).mph" `
  -Config "external\comsol_icp_export\config\icp_cf4_o2_v20.json" `
  -OutDir "_external_exports\icp_cf4_o2_v20"
```

The Java step verifies the configured dataset-to-solution and solution-to-study
references, then checks that `solution_number` contains the configured parameter
value. It reads the saved solution without changing model parameters or solving
the study. The checked-in V20 profile therefore selects solution 1 of
`std2`/`sol2` through `dset3`. It then writes `mesh.mphtxt`,
`field_samples.csv`, and `export_manifest.json`.
The manifest records COMSOL version, solution number, expressions, units,
vacuum domains, and SHA-256 hashes for the model, config, mesh, and field
samples. Create and review two explicit SI inputs before packing:

- a canonical release table (`particles.csv` columns, including `r_m`, `z_m`,
  `vr_mps`, `vz_mps`, `mass_kg`, and `drag_diameter_m`);
- one complete `boundaries.csv` covering every exported part with its wall law
  and material metadata.

Then use the generic profile:

```powershell
particle-tracer comsol build-case --profile icp_cf4_o2 `
  --raw-export-dir "_external_exports\icp_cf4_o2_v20" `
  --release-table "path\to\particles.csv" `
  --boundaries "path\to\boundaries.csv" `
  --out-dir "_external_exports\icp_case_v02" `
  --diagnostic-grid-spacing-m 5e-4 `
  --dt-s 2e-8 --t-end-s 2e-6 `
  --drag-law stokes --force electric `
  --gas-dynamic-viscosity-Pas 1.8e-5 `
  --release-inward-offset-m 1e-8 `
  --release-projection-tolerance-m 1e-10
```

With `--raw-export-dir`, model provenance, coordinate scale, solution number,
and vacuum domains come only from `export_manifest.json`; duplicate CLI values
are rejected. Diagnostic grid spacing and runnable solver `dt`/`t_end` are
case choices and must still be supplied explicitly; the packer never inserts
physical or temporal defaults. Without `--raw-export-dir`,
`--coordinate-scale-m-per-model-unit` is also mandatory. Geometry-only packing
does not require unused solver time values. The profile converts the exported `(r,z)` grid to SI, preserves
electric field as `E` (never fixed-reference acceleration), and emits a
schema-v2 COMSOL manifest plus the strict canonical run config. Physical walls
come from the explicitly selected vacuum-domain mesh boundary. The field
`valid_mask` is retained solely as field support and cannot become staircase
geometry. The workflow does not generate particles, guess wall roles, or
provide fallback materials.


## Mesh-node samples

The exporter writes two sample tables from one evaluation pass.

- `field_samples.csv` — the configured `r`/`z` grid, kept as a readable
  reference of the same export.
- `field_samples_nodes.csv` — every configured expression evaluated at the
  COMSOL mesh vertex coordinates, keyed by `node_index`.  That index is the
  row order of `mesh.mphtxt`'s vertex block, so the case builder joins node
  values to mesh topology exactly, without coordinate rounding.

`export_manifest.json` declares `field_node_samples_sha256`,
`field_node_sample_count`, and `field_node_identity`.  `particle-tracer comsol
build-case --raw-export-dir` builds a mesh-native case when that digest is
present, and a resampled grid case otherwise.  Presence of the file on disk is
never sufficient on its own.

Every `Interp` feature is restricted to `vacuum_domain_ids`, so a mesh vertex
lying on a vacuum/solid interface returns its vacuum-side value instead of a
`NaN` from a domain where the expression is undefined.  `vacuum_domain_ids`
must therefore be non-empty in the exporter configuration.
