# particle_tracer_unified

Particle trajectory solver for supplied 2D/3D field data and explicit boundary
tables.

## Read First

- `docs/architecture.md`: package layout and extension rules
- `docs/canonical_input_bundle.md`: canonical adapter boundary for COMSOL and
  non-COMSOL model sources
- `docs/comsol_parity.md`: COMSOL faithful comparison design
- `docs/comsol_onboarding.md`: short COMSOL export and comparison runbook
- `docs/numerics_contract.md`: continuous model and integrator notes
- `docs/productization/sim_rev3/vv/README.md`: V&V workflow for sampled,
  full-reference, and sharded comparisons
- `docs/release_packaging_policy.md`: documentation, template, and large-asset
  policy for release candidates
- `examples/minimal_surface_release_production/README.md`: COMSOL-free
  production surface-release quickstart
- `docs/visualization_workflow.md`: compact graph and optional animation
  workflow

## Install

```powershell
py -3 -m pip install -e .
```

## Run

COMSOL-free production surface-release quickstart:

```powershell
py -3 run_from_yaml.py examples/minimal_surface_release_production/run_config.yaml --check-input --output-dir _out_surface_release_check
py -3 run_from_yaml.py examples/minimal_surface_release_production/run_config.yaml --output-dir _out_surface_release
```

Small generic examples:

```powershell
py -3 run_from_yaml.py examples/minimal_2d/run_config.yaml --output-dir _out_minimal_2d
py -3 run_from_yaml.py examples/minimal_3d/run_config.yaml --output-dir _out_minimal_3d
```

After installation, the console script is also available:

```powershell
particle-tracer-run examples/minimal_2d/run_config.yaml --output-dir _out_minimal_2d
```

Default non-faithful runs use compact `output.mode: standard`: final particles,
solver report, wall/coating summaries, and compact prepare reports. Use
`output.mode: debug` or explicit `output.write_*` flags when a comparison needs
deep artifacts such as trajectories, `wall_events.csv`,
`runtime_step_summary.csv`, `collision_diagnostics.json`, or
`force_contributions.csv`. Legacy `output.artifact_mode: full` maps to debug.
When trajectory output is enabled, `output.save_every` controls saved snapshot
cadence; if omitted, the legacy `solver.save_every` value is used.

## Inputs

Normal cases are built from `run_config.yaml` plus explicit tables:

- `particles_csv`: initial particles
- `materials_csv`: optional material table
- `part_walls_csv`: optional wall behavior table
- `source_events_csv`: optional source timing/gain events
- `process_steps_csv`: optional time labels
- provider files such as precomputed geometry and field `.npz` bundles

Run a case-level input check before production comparisons:

```powershell
py -3 run_from_yaml.py <case>/run_config.yaml --check-input --output-dir <out_check>
```

The check writes provider and particle preflight reports. Fix exported inputs
before adding solver-side workarounds.

For chamber-part release studies, surface particles may intentionally start on
the wall. Enable `source.preprocess.boundary_release` to classify those release
points against the explicit boundary, offset them into the simulated domain,
and then run the usual preflight. Keep COMSOL faithful cases strict; use the
surface-release path for production flake/resuspension trajectories. The solver
does not infer fracture, deposit failure, or release populations; provide those
as particle/source inputs and use preflight reports to verify the numerical
initial condition.

## COMSOL Workflow

`comsol_faithful` mode is manifest-first. The solver package does not read
`.mph` files; COMSOL API/export code belongs under `external/` or `tools/`.

```yaml
mode: comsol_faithful
comsol:
  manifest: comsol_case_manifest.yaml
```

The runtime integration lives in `particle_tracer_unified/io/comsol.py`. COMSOL
release particles, boundary maps, wall laws, coordinate scale, and force
inventory are loaded from the manifest and validated before the solver runs.

Comparison entry points:

```powershell
particle-tracer-field-compare --help
particle-tracer-acceleration-compare --help
particle-tracer-trajectory-compare --help
particle-tracer-boundary-compare --help
particle-tracer-first-step-compare --help
particle-tracer-comsol-full-diagnostics --help
particle-tracer-build-comsol-case --help
particle-tracer-compare-reference --help
particle-tracer-collect-summaries --help
particle-tracer-validate-artifacts --help
particle-tracer-residual-gap-summary --help
```

For sampled/full/sharded validation roots, check that the compact artifacts are
present before running residual diagnostics:

```powershell
particle-tracer-validate-artifacts <root> --workflow sampled
particle-tracer-validate-artifacts <root> --workflow sharded --require-source-diagnostics
```

Canonical productization templates live under
`docs/productization/sim_rev3/templates/`, including the V&V acceptance matrix,
task record template, COMSOL minimal manifest template, compare summary schema,
and first-step/force contribution CSV schemas.

## Visualization

Graphs are safe from compact standard outputs:

```powershell
particle-tracer-export-visualizations --output-dir <out> --modules graphs
```

Animations are optional and should be generated only from debug or explicit
trajectory-saving runs. Use `--animation-max-particles`,
`--animation-max-frames`, and `--skip-all-particles-animation` for large cases.
See `docs/visualization_workflow.md`.

## Development Checks

Fast smoke check:

```powershell
py -3 -m pytest -q tests/smoke_test.py
```

Focused COMSOL parity checks:

```powershell
py -3 -m pytest -q tests/test_comsol_manifest.py tests/test_comsol_release_reader.py tests/test_comsol_boundary_reader.py tests/test_comsol_faithful_runtime.py
```

## Scope

Keep the core package centered on provider-backed fields, explicit geometry,
particle tables, boundary behavior, and solver execution through
`PreparedRuntime`. Prefer small modules with clear inputs over broad validation
layers, case-specific branches, or duplicated checks.
