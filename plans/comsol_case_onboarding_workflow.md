# COMSOL Case Onboarding Workflow

Status: Codex working guide, not an implemented solver feature.

Use this file when adding or evaluating a new COMSOL-derived case. The goal is
to turn a COMSOL model into a clean solver input problem without adding
case-specific rescue logic to the solver.

## Purpose

For each COMSOL case, make the following explicit before production runs:

- what geometry and boundary parts are physically relevant
- which regions have a valid medium and field support
- which force components are exported and how units are converted
- where particles are released and what physical particle model is used
- what `dt`, `t_end`, output cadence, and particle count mean physically
- which checks prove the exported case is valid before time integration

This workflow is deliberately outside the solver hot path. COMSOL extraction
and case packing live under `external/` or case-generation scripts. The solver
package consumes only solver-ready `run_config.yaml`, CSV, and `.npz` inputs.

## Directory Policy

- Keep source `.mph` files under `data/`.
- Keep COMSOL API code under `external/<exporter_name>/`.
- Keep raw export outputs under `_external_exports/<case_name>/`.
- Keep solver-ready cases under `examples/<case_name>/`.
- Keep run outputs under `_out_<case_name>/`.
- Keep report figures under `report_assets/<case_name>/` only when they are
  meant to be shared in a report.
- Do not commit crash logs, local cache folders, or root-level duplicate `.mph`
  files.

## Non-Negotiable Rules

- Do not import COMSOL libraries from `particle_tracer_unified`.
- Do not fill missing field regions inside the runtime solver.
- Do not move initial particles inside the solver to make contracts pass.
- Do not tune wall-hit retry counts as an accuracy strategy.
- Do not copy `dt`, `t_end`, or particle settings from another case without a
  physical time-scale check.
- Do not treat lower `stuck`, `invalid_mask`, or `numerical_boundary_stopped`
  counts as success unless the provider and boundary contracts are correct.
- Do not add process-recipe behavior to solve field/boundary problems.

## Case Intake Checklist

Before writing code, record these points in the case manifest or a short case
summary file:

- COMSOL model path, file hash, COMSOL version, study, dataset, and parameter
  values.
- Coordinate system and unit scale, for example model centimetres to SI metres.
- Geometry dimension: 2D planar, 2D axisymmetric r-z, or true 3D.
- Domain IDs and boundary IDs, including regions that have geometry but no
  valid medium.
- Physical role of important parts: wafer, RF electrode, sidewall, chamber wall,
  source surface, symmetry axis, outlet, or inactive solid.
- Required field quantities and units.
- Particle material, diameter, density, mass, charge model, and initial velocity
  model.
- Wall law for each physical boundary class: stick, reflect, absorb, react, or
  remove.
- Intended run tier: smoke, 10k production, full visualization, or report run.

Stop and ask the user when any of these are ambiguous:

- boundary part identity affects physical conclusions
- coordinate scale or unit conversion is uncertain
- wall law or source surface is guessed from appearance only
- particle charge/mass model is not physically specified
- a COMSOL region touches the calculation domain but has no valid medium
- time-dependent fields or moving geometry require a new provider contract

## Export Inventory

Run expression and model inventory before exporting field bundles. The exporter
must write:

- `export_manifest.json`
- `expression_inventory.json`
- mesh or geometry export, for example `mesh.mphtxt`
- raw sampled field table or equivalent structured data

The manifest should include:

- model path and hash
- COMSOL version
- study and dataset
- parameter values
- coordinate scale
- expression map
- units
- grid shape or mesh size
- valid-medium/support definition
- electric-field quantity names and units; electric force is computed in the solver from current particle `q(t)/m`

Required solver-facing field quantities are:

- velocity: `ux`, `uy` or `ux`, `uy`, `uz`
- dynamic viscosity: `mu`
- electric field: `E_x`, `E_y` or `E_x`, `E_y`, `E_z`
- `valid_mask`
- axes and, when applicable, `times`

COMSOL-only diagnostics such as electric potential, electron
density, ion density, temperature, pressure, or ion flux should be exported for
review, but they must not be required by the solver hot path unless a new
provider contract is intentionally designed.

## Solver Case Packing

Each solver-ready case should contain:

- `run_config.yaml`
- `materials.csv`
- `part_walls.csv`
- `particles.csv`
- `generated/comsol_geometry_*.npz`
- `generated/comsol_field_*.npz`
- a case summary JSON when generation choices are nontrivial

Standard production configs should enable:

```yaml
provider_contract:
  boundary_field_support: strict
input_contract:
  initial_particle_field_support: strict
output:
  artifact_mode: minimal
```

Use `artifact_mode: full` only for report-quality visualization runs or manual
debugging.

## Validation Before Solving

Run the provider/input contract check first:

```powershell
py -3 run_from_yaml.py examples/<case_name>/run_config.yaml `
  --check-input `
  --output-dir _out_<case_name>_check
```

Inspect:

- `provider_contract_report.json`
- `provider_boundary_summary.csv`
- `provider_boundary_violations.csv`
- `input_contract_report.json`
- `input_particle_violations.csv`

Failure meaning:

- Provider failure: field export or boundary support is wrong.
- Input failure: particle generation/release-domain selection is wrong.
- Both fail: fix export and geometry interpretation before particle tuning.

Do not proceed to production trajectories until strict contracts pass or the
user explicitly accepts a known experimental case.

## Physical Time-Scale Selection

Choose `dt` and `t_end` from case physics, not from old benchmark defaults.

Useful scales:

```text
tau_p = rho_p * d_p^2 / (18 * mu)
a_E = q * E / m
t_cross = L / max(|u|, eps)
t_accel = sqrt(2 * L / max(|a|, eps))
t_rf = 1 / f_rf
```

Guidance:

- `dt` should resolve the smallest active response scale that affects motion.
- `t_end` should cover the expected physical flight or residence time.
- For time-dependent fields, field sampling times must cover all requested
  integrator stage times.
- Animation frame spacing can be coarser than solver `dt`.

Record the selected values and the reason in the case summary or report.

## Run Tiers

Smoke run:

```powershell
py -3 run_from_yaml.py examples/<case_name>/run_config.yaml `
  --output-dir _out_<case_name>_smoke
```

Use 500-1000 particles, short physics time, and minimal artifacts.

Production run:

```powershell
py -3 run_from_yaml.py examples/<case_name>/run_config.yaml `
  --output-dir _out_<case_name>_10k
```

Use the intended 10k particle count, physically selected `dt`/`t_end`, and
minimal artifacts.

Visualization/report run:

- reuse the same physical settings when possible
- enable full artifacts only when needed
- copy selected graphs into `report_assets/<case_name>/`

## Visualization Checklist

For interpretation, prefer a small set of high-value plots:

- geometry with all relevant part outlines, including non-fluid parts when they
  explain the device structure
- a part-ID or boundary-ID view when identity matters
- SDF and valid field support view
- total force or acceleration magnitude
- force components by physical type when available
- flow speed and vectors over geometry
- initial particle source positions
- final particle states by part
- representative trajectories
- animation for report or user review

If a region is white or missing in a field plot, determine whether it is:

- outside the valid medium
- inside a solid or inactive part
- outside the sampled field bundle
- missing because expression export failed

Do not hide missing field regions with visual interpolation unless the plot is
clearly labelled as presentation-only.

## Production Acceptance

A production case should satisfy:

- provider contract passed
- input contract passed
- `invalid_mask_stopped_count == 0`
- `numerical_boundary_stopped_count == 0`
- `unresolved_crossing_count == 0`
- `boundary_event_contract_passed == 1` when that report field exists
- no nonfinite final positions or velocities
- runtime and memory are acceptable for the selected particle count
- visual plots match the known COMSOL geometry and field structure

If these fail, classify the failure before changing code:

| Symptom | First place to investigate |
| --- | --- |
| provider contract fails | field export, coordinate scale, valid support, boundary IDs |
| input contract fails | particle generator, source surface, release-domain support |
| invalid mask during solve | time support, acceleration/velocity field support, stage sampling |
| numerical boundary stops | boundary primitives, hit-time solve, part identity |
| many stuck particles | wall law, contact state, source direction, boundary geometry |
| unexpected deposition parts | wall classification, source velocity, particle charge/mass |
| blank field regions | medium support, solid regions, failed expressions |

## What To Avoid Reintroducing

- old fixed 10k COMSOL configs as universal gates
- mesh production paths unless true mesh field data is available
- solver-side field regularization or clipping
- particle push-off or release-position rescue in runtime code
- long debug-output directories committed as source
- tests that only preserve old implementation detail and do not protect a
  numerical or user-facing contract

## Codex Working Loop

For future work, follow this order:

1. Read this file and the target case summary.
2. Inspect current `git status` and avoid reverting unrelated user changes.
3. Identify whether the task is export, packing, provider contract, input
   contract, solver behavior, visualization, or reporting.
4. Make the smallest change that resolves that category.
5. Run the relevant contract or smoke command.
6. Summarize whether the result is physically valid, not just numerically
   quieter.
