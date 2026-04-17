# particle_tracer_unified

A runnable particle trajectory package for supplied 2D/3D fields and explicit boundary truth.

## What this package currently does

- reads `run_config.yaml`
- builds a `PreparedRuntime`
- preprocesses particles with material-aware source laws
- supports optional source events and lightweight process-step time labels
- samples supplied steady or time-dependent velocity, electric-field, and gas-property fields
- runs 2D or 3D synthetic-box examples through a practical high-fidelity solver
- uses Numba free-flight kernels and geometry-aware wall-shear sampling
- ships a fully runnable `examples/minimal_2d` and `examples/minimal_3d`

## Install

```bash
pip install -e .
```

## Run a 2D example

```bash
python run_from_yaml.py examples/minimal_2d/run_config.yaml --output-dir out_2d
```

After `pip install -e .`, the same entry point is available as:

```bash
particle-tracer-run examples/minimal_2d/run_config.yaml --output-dir out_2d
```

## Run a 3D example

```bash
python run_from_yaml.py examples/minimal_3d/run_config.yaml --output-dir out_3d
```

## Build and check COMSOL-derived 2D data

The repository includes `data/argon_gec_ccp_base2.mphtxt`.
Run the builder to extract COMSOL geometry and the rectilinear compatibility field package:

```bash
py -3 tools/build_comsol_case.py \
  --field-bundle data/regridded_repo_field_bundle_argon_gec_ccp_base2_2d.npz
```

Then check the generated provider/input contracts:

```bash
python run_from_yaml.py examples/comsol_from_data_2d/run_config.yaml --check-input --output-dir out_comsol_2d_check
```

Notes:
- geometry is extracted from COMSOL mesh (`edg`/`quad`) into `generated/comsol_geometry_2d.npz`
- rectilinear compatibility field support is written to `generated/comsol_field_2d.npz`
- the bundled rectilinear COMSOL field bundle is exported with finite field support plus ghost cells
- do not run production trajectories until both provider and input contracts pass

## Provider And Input Contracts

For supplied field data, the field provider must first cover the explicit boundary support domain. A physically valid wall is not enough if the field cannot be sampled cleanly just inside that wall. For rectilinear data this means the boundary-adjacent interpolation stencil must be valid.

Particles must also start inside the **clean field sample domain**. Being physically inside the geometry is not enough for a rectilinear field: bilinear/trilinear interpolation also needs a fully valid stencil.

The field support mask is owned by the field provider and is not clipped to the geometry mask at runtime. This allows a valid export bundle to include ghost/support nodes needed for interpolation near walls while geometry still controls boundary detection.
The COMSOL builder derives the exported field support mask from finite field quantities, not from the geometry domain mask. It also adds a finite ghost-cell band by edge extrapolation, defaulting to 8 cells, so boundary handling is not limited by a grid that ends exactly at the wall.
Precomputed providers validate axes, times, active-support values, and mesh triangle topology before the solver runs.

Check a case before solving:

```bash
python run_from_yaml.py <case>/run_config.yaml --check-input --output-dir <out_check>
```

This writes:
- `provider_contract_report.json`
- `provider_boundary_summary.csv` with per-part failure counts and boundary/offset bounding boxes
- `provider_boundary_violations.csv` with every sampled non-clean boundary support point and numeric coordinate columns
- `input_contract_report.json`
- `input_particle_violations.csv` when non-clean initial particles exist

With `provider_contract.boundary_field_support: strict`, normal solver runs stop before time integration if the field/boundary provider pair is not compatible. This is intentional: boundary-adjacent field gaps should be fixed in the export bundle or by using a mesh-native field provider, not hidden by trajectory rescue logic.

With `input_contract.initial_particle_field_support: strict`, normal solver runs stop before time integration if any initial particle is `mixed_stencil` or `hard_invalid`. This is intentional: invalid release-domain data should be fixed at input generation time, not hidden by trajectory rescue logic. If the provider check fails, fix the field/boundary export first. If only the input check fails, regenerate or correct `particles.csv` before running production comparisons.

## COMSOL Case Production Check

For a COMSOL-derived 2D case whose provider and input contracts pass, a conservative trajectory profile is:

- `input_contract.initial_particle_field_support: strict`
- `provider_contract.boundary_field_support: strict`
- `solver.integrator: etd2`
- `solver.adaptive_substep_enabled: 0`
- `solver.min_remaining_dt_ratio: 0.0`
- `solver.valid_mask_policy: retry_then_stop`
- `output.artifact_mode: minimal` with `output.write_collision_diagnostics: 1`

Run:

```bash
python run_from_yaml.py <case>/run_config.yaml --output-dir <out_run>
```

Check `solver_report.json` and `collision_diagnostics.json` for residual numerical behavior:
- `numerical_boundary_stopped_count`
- `unresolved_crossing_count`
- `nearest_projection_fallback_count`
- `boundary_event_contract_passed`
- `valid_mask_violation_count`
- `valid_mask_violation_particle_count`

`boundary_event_contract_passed` must be `1` for production runs. `max_hits_reached_count` and `nearest_projection_fallback_count` are not accuracy tuning targets. If either appears in production, check the provider/boundary contract and boundary-event handling.

The daily reference-vs-candidate gate should be run as a lightweight check after provider contracts pass. Generated override configs are written under the output directory:

```bash
python tools/compare_against_reference.py \
  --reference-config <reference_case>/run_config.yaml \
  --run candidate=<candidate_case>/run_config.yaml \
  --output-root <out_compare> \
  --override-t-end <short_physics_window_s> \
  --artifact-mode minimal \
  --per-run-timeout-s 300
```

This tool writes one timestamped summary containing:
- `class_match_ratio_vs_reference`
- `class_transition_summary_vs_reference`
- `geometry_feature_delta_vs_reference`: SDF, nearest-boundary-distance, final-position, speed, and nearest-part transition deltas on the reference geometry
- `numerical_boundary_stopped_count`
- `unresolved_crossing_count`
- `nearest_projection_fallback_count`
- `boundary_event_failure_count`
- `runtime_s`
- `solver_core_s`, `solver_step_loop_s`, and estimated NumPy buffer bytes
- `field_backend_kind`
- pairwise deltas when exactly two candidate runs are provided

For manual full-reference checks, run the same command without `--override-t-end` and with `--artifact-mode full` or no artifact override. That path is intentionally heavier than the daily gate.

## Run the smoke test

```bash
python -m pytest -q tests/smoke_test.py
```

## Main concepts

- `PreparedRuntime`: the canonical solver input built from `run_config.yaml`.
- Required case files: `materials.csv`, `part_walls.csv`, `particles.csv`, plus the configured geometry and field provider data.
- Contract reports: `provider_contract_report.json` and `input_contract_report.json` separate bad exported data from solver failures before time integration.
- Production diagnostics: `solver_report.json`, `collision_diagnostics.json`, `final_particles.csv`, and `coating_summary_by_part.csv`.
- Field extension point: add or improve providers behind the same `FieldProviderND` and `GeometryProviderND` data contracts.
- Solver extension point: keep new integration or boundary behavior behind the existing prepared-runtime entry point instead of adding case-specific rescue paths.
- Optional Brownian/Langevin motion is disabled by default and can be enabled with `solver.stochastic_motion.enabled: true`.
- Optional 2D charge evolution is disabled by default and can be enabled with `solver.charge_model.enabled: true`.
  - `mode: te_relaxation` (`v1`) relaxes charge from local electron temperature.
  - `mode: density_temperature_flux_relaxation` (`v2`) uses local density/temperature flux balance.
  - Both modes use scalar background distributions plus `E_x/E_y`; COMSOL flux vectors are not required.
- For `solver.drag_model: epstein`, rectilinear COMSOL field quantities `rho_g` and `T` are sampled for low-pressure drag when present. The scalar `gas.*` values remain fallbacks, field `mu` can be reported, and `p` is kept diagnostic rather than used directly by drag.
- Optional overlays: `process_steps.csv` is only a time-label overlay, and `source_events.csv` is only for explicit source timing/gain events.

## Numerics contract

The current continuous-model and discrete-integrator contract is summarized in:

```text
docs/numerics_contract.md
```

The active implementation guide for the current boundary-performance work is:

```text
plans/boundary_performance_plan.md
```

The COMSOL case onboarding checklist for future model imports is:

```text
plans/comsol_case_onboarding_workflow.md
```

## Included examples

- `examples/minimal_2d`
- `examples/minimal_3d`

Each example includes all files needed to run directly.

## Scratch outputs

- `_tmp_*`, `_out_*`, `demo_output/`, and example `run_output*` directories are treated as local scratch outputs and are not source-controlled.
- Timestamped comparison folders under `demo_output/reference_compare` are local benchmark artifacts.
- `solver_report.json` and `runtime_step_summary.csv` include `valid_mask` diagnostics plus `field_backend_kind`, and opt-in runs can also report `invalid_mask_stopped` particles.
- `tools/compare_against_reference.py --override-t-end ... --artifact-mode minimal` is the standard lightweight benchmark gate.

## Current scope

This package is focused on supplied 2D/3D fields and explicit boundary truth. Higher-level process recipes are outside the current solver scope; the process CSV is only a time-label overlay. The core extension path is provider-backed field data plus focused solver modules behind the same `PreparedRuntime` entry point.
