# COMSOL Case Handoff

Use this when a new COMSOL `.mph` model must be exported, packed into a
solver-ready case, run, and compared with COMSOL Particle Tracing.

For the solver-side design contract, read `docs/comsol_parity.md`. This file is
only the operational handoff checklist.

## Rule

Keep COMSOL API code outside `particle_tracer_unified`. The solver package
should consume explicit `run_config.yaml`, CSV, and NPZ files only.

Do not add solver-side rescue logic to hide bad export data. Classify failures
as export, coordinate scale, field support, release setup, force model, wall
law, or numerical setting issues before changing solver code.

## Ask First

Collect these before judging results:

- `.mph` path and COMSOL version
- study, dataset, parameter point, and time/solution index
- coordinate system and coordinate scale to SI meters
- particle material, diameter, density, mass, charge or charge model
- source/release feature identity and release timing
- enabled Particle Tracing force nodes
- intended wall law for each relevant boundary group
- COMSOL particle result CSV when exact comparison is required
- COMSOL release table when release positions/times are not trivial

Stop and ask when coordinate scale, source surface, release timing, charge, or
wall laws are ambiguous.

## Paths

- source model: `data/<case_name>.mph`
- raw export: `_external_exports/<case_name>/`
- solver case: `examples/<case_name>/`
- solver output: `_out_<case_name>_<run_label>/`

Large raw exports and solver outputs should stay local unless the user asks for
a report bundle.

## Export

Run the generic exporter or a case-specific external exporter:

```powershell
.\external\comsol_particle_export\run_export.ps1 `
  -ComsolExe "C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64\comsolbatch.exe" `
  -Mph "data\<case_name>.mph" `
  -Config "external\comsol_particle_export\config\<case_config>.json" `
  -OutDir "_external_exports\<case_name>"
```

Expected raw artifacts usually include inventories, export manifest, mesh data,
and field samples. Do not hard-code model-specific COMSOL variable names inside
the solver package.

## Validate Raw Export

```powershell
py -3 external\comsol_particle_export\validate_export.py `
  --raw-export-dir "_external_exports\<case_name>" `
  --config "external\comsol_particle_export\config\<case_config>.json" `
  --summary-out "_external_exports\<case_name>\raw_export_validation.json"
```

Fix export/config problems before building a solver case.

## Build Solver Case

A solver-ready case should contain:

- `run_config.yaml`
- `materials.csv`
- `part_walls.csv`
- `particles.csv`
- generated geometry/field NPZ files

Use an existing packer only when its geometry, units, release, and wall
assumptions match the COMSOL model. Otherwise add or adapt a packer under
`external/<case_exporter>/`.

For the existing ICP bridge:

```powershell
py -3 -m external.comsol_icp_export.comsol_icp_export.pack_solver_case `
  --raw-export-dir "_external_exports\<case_name>" `
  --out-dir "examples\<case_name>" `
  --particle-count 1000
```

## Preflight And Run

```powershell
py -3 run_from_yaml.py examples\<case_name>\run_config.yaml `
  --prepare-only `
  --output-dir "_out_<case_name>_prepare"
```

Check particle release, material/wall mapping, field support, force catalog, and
preflight reports before a production run.

Use a short smoke run first. Scale particle count and `t_end` only after:

- provider/input checks pass
- release direction and timing look correct
- field signs/scales are plausible
- wall outcomes match intended laws
- invalid-mask and numerical-boundary counts are not hiding export mistakes

## Compare

When COMSOL particle results are available:

```powershell
py -3 external\comsol_particle_export\compare_particle_results.py `
  --solver-output-dir "_out_<case_name>_<run_label>" `
  --comsol-csv "_external_exports\<case_name>\comsol_particle_results.csv" `
  --out-dir "_out_<case_name>_<run_label>\comsol_compare"
```

For faithful runtime comparison, use:

```powershell
particle-tracer-field-compare --help
particle-tracer-acceleration-compare --help
particle-tracer-trajectory-compare --help
particle-tracer-boundary-compare --help
```

## Failure Triage

- Geometry/scale: wrong coordinate system or meters-per-model-unit scale.
- Field support: missing finite values, wrong expression, wrong sign, or
  boundary-adjacent support gaps.
- Release: wrong source entity, count, timing, initial velocity, or weighting.
- Force model: enabled force inventory does not match COMSOL.
- Wall laws: boundary IDs or wall interactions are mapped incorrectly.
- Numerics: time step, collision replay, or output cadence differs.

Fix the earliest bad artifact in this chain. Avoid compensating for upstream
export mistakes in the solver core.
