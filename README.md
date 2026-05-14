# particle_tracer_unified

Particle trajectory solver for supplied 2D/3D field data and explicit boundary
tables.

## Read First

- `docs/architecture.md`: package layout and extension rules
- `docs/comsol_parity.md`: COMSOL faithful comparison design
- `COMSOL_COPILOT_HANDOFF.md`: operational checklist for importing a new COMSOL model
- `docs/numerics_contract.md`: continuous model and integrator notes

## Install

```powershell
py -3 -m pip install -e .
```

## Run

```powershell
py -3 run_from_yaml.py examples/minimal_2d/run_config.yaml --output-dir _out_minimal_2d
py -3 run_from_yaml.py examples/minimal_3d/run_config.yaml --output-dir _out_minimal_3d
```

After installation, the console script is also available:

```powershell
particle-tracer-run examples/minimal_2d/run_config.yaml --output-dir _out_minimal_2d
```

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
```

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
