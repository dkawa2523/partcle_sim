# Minimal Surface-Release Production Example

This COMSOL-free case shows the production path for particles that originate on
or just outside a chamber surface. It uses synthetic box geometry and a steady
linear-shear field so the case stays small and quick.

## Input Check

```powershell
py -3 run_from_yaml.py examples/minimal_surface_release_production/run_config.yaml --check-input --output-dir _out_surface_release_check
```

Expected check artifacts:

- `prepared_runtime_summary.json`
- `provider_contract_report.json`
- `input_contract_report.json`
- `source_particle_diagnostics.csv`

## Run

```powershell
py -3 run_from_yaml.py examples/minimal_surface_release_production/run_config.yaml --output-dir _out_surface_release
```

Expected standard artifacts:

- `final_particles.csv`
- `solver_report.json`
- `prepared_runtime_summary.json`
- `provider_contract_report.json`
- `input_contract_report.json`
- `wall_summary.json`
- `wall_summary_by_part.csv`
- `coating_summary.json`
- `coating_summary_by_part.csv`

Standard mode does not write trajectory arrays, wall-event CSVs, runtime step
summaries, collision diagnostics, or force contributions.

## Surface Release Settings

`source.preprocess.boundary_release: true` classifies raw release coordinates
against the explicit box boundary before the input preflight. In this case the
raw particles start near `part_id=101` (`release_wall`).

`boundary_capture_tolerance_m: 5.0e-4` is the near-boundary classification
window. It can be larger than the inward displacement to catch exported or
off-grid release points.

`boundary_inward_offset_m: 2.0e-6` is the small displacement applied after a
release point has been classified and projected. It should stay small relative
to geometry scale.

`part_walls.csv` gives the release wall a `stick` wall law. Particle `1`
initially points back toward that wall after the inward release offset, so the
standard run has a simple stuck-wall outcome to inspect in `wall_summary.json`.

## Debug Artifacts

Use debug output only when investigating a comparison or wall/source issue:

```yaml
output:
  mode: debug
```

Then rerun the same command. Debug mode writes deeper artifacts such as
`positions_2d.npy`, `save_frames.csv`, `wall_events.csv`,
`runtime_step_summary.csv`, `source_particle_diagnostics.csv`,
`collision_diagnostics.json`, and `force_contributions.csv`.
