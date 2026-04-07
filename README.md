# particle_tracer_unified v29

A self-contained, runnable stabilization release for step-aware particle trajectory simulation.

This release fixes the missing example/input problem in the prior packaging by **including a complete minimal 2D and 3D dataset** and by verifying the canonical CLI path with smoke tests.

## What this package currently does

- reads `run_config.yaml`
- builds a `PreparedRuntime`
- preprocesses particles with material-aware source laws
- binds source events to process steps or recipe transitions
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

## Run a 3D example

```bash
python run_from_yaml.py examples/minimal_3d/run_config.yaml --output-dir out_3d
```

## Build and run COMSOL-derived 2D case

The repository includes `data/argon_gec_ccp_base2.mphtxt`.  
Run the builder to extract COMSOL geometry plus both field backends:

```bash
py -3 tools/build_comsol_case.py
```

Then run trajectories using the generated rectilinear compatibility path:

```bash
python run_from_yaml.py examples/comsol_from_data_2d/run_config.yaml --output-dir out_comsol_2d
```

Or try the 2D triangle-mesh backend directly:

```bash
python run_from_yaml.py examples/comsol_from_data_2d/run_config_mesh.yaml --output-dir out_comsol_2d_mesh
```

Notes:
- geometry is extracted from COMSOL mesh (`edg`/`quad`) into `generated/comsol_geometry_2d.npz`
- rectilinear compatibility field support is written to `generated/comsol_field_2d.npz`
- 2D triangle-mesh field support is written to `generated/comsol_field_mesh_2d.npz`

## ETD2 stabilization profile (opt-in)

For the 10k COMSOL-derived 2D production case, an ETD2 stability-focused profile is included:

```bash
python run_from_yaml.py examples/comsol_from_data_2d_10k/run_config_prod_etd2_stable.yaml --output-dir out_comsol_2d_etd2_stable
```

This profile keeps backward-compatible defaults in code and opts in via config:
- `solver.integrator: etd2`
- `solver.max_hits_retry_splits: 2`
- `solver.max_hits_retry_local_adaptive_enabled: 1`
- `solver.adaptive_substep_enabled: 0` (global OFF, retry-local adaptive ON)
- `solver.min_remaining_dt_ratio: 0.0`

Check `collision_diagnostics.json` for ETD2 stability behavior:
- `max_hits_retry_count`
- `max_hits_retry_exhausted_count`
- `dropped_remaining_dt_total_s`
- `max_hits_reached_count`
- `unresolved_crossing_count`
- `valid_mask_violation_count`
- `valid_mask_violation_particle_count`

You can run a manual stability/performance gate:

```bash
python tools/check_stability_profile.py \
  --base-config examples/comsol_from_data_2d_10k/run_config_prod_etd2_base.yaml \
  --candidate-config examples/comsol_from_data_2d_10k/run_config_prod_etd2_stable.yaml \
  --max-runtime-increase-ratio 0.20 \
  --min-max-hits-reduction-ratio 0.30
```

The canonical reference-vs-candidate comparison tool is:

```bash
python tools/compare_against_reference.py \
  --reference-config examples/comsol_from_data_2d_10k/run_config_eval_ref_etd2.yaml \
  --run etd2_base=examples/comsol_from_data_2d_10k/run_config_prod_etd2_base.yaml \
  --run etd2_stable=examples/comsol_from_data_2d_10k/run_config_prod_etd2_stable.yaml \
  --output-root demo_output/reference_compare
```

This tool writes one timestamped summary containing:
- `class_match_ratio_vs_reference`
- `unresolved_crossing_count`
- `runtime_s`
- `field_backend_kind`
- pairwise deltas when exactly two candidate runs are provided

For `valid_mask` rollout decisions, the canonical policy gate is:

```bash
python tools/evaluate_valid_mask_rollout.py \
  --config examples/comsol_from_data_2d_10k/run_config_prod_etd2_stable.yaml \
  --reference-config examples/comsol_from_data_2d_10k/run_config_eval_ref_etd2.yaml \
  --output-root demo_output/valid_mask_rollout \
  --max-runtime-increase-ratio 0.10 \
  --min-class-match-ratio 0.9845
```

This writes a timestamped rollout summary with:
- `diagnostic` vs `retry_then_stop`
- `runtime_increase_ratio`
- `class_match_ratio_vs_reference`
- `invalid_mask_stopped_count_delta`
- a simple `rollout_recommendation`

## Run the smoke test

```bash
python -m pytest -q tests/smoke_test.py
```

## Main concepts

- `PreparedRuntime`: the canonical solver input.
- `process_steps.csv` or `recipe_manifest.yaml`: time segmentation and step-aware control.
- `source_events.csv`: burst / gate / gain events tied to steps or transitions.
- `materials.csv`, `part_walls.csv`, `particles.csv`: source/material definitions and particle inputs.
- `IntegratorSpec`: the internal contract for `drag_relaxation`, `etd`, and `etd2`.
- `BoundaryService`: the internal contract for geometry truth queries and boundary hits.
- `solver.valid_mask_policy`: `diagnostic` by default, or opt-in `retry_then_stop` for authoritative stopping on invalid field regions.
- `providers.field.kind`: rectilinear compatibility (`precomputed_npz`) or 2D triangle-mesh field support (`precomputed_triangle_mesh_npz`).
- `high_fidelity_common.py`: a thin compatibility hub that re-exports the shared solver entry.
- `high_fidelity_freeflight.py`, `high_fidelity_collision.py`, `high_fidelity_runtime.py`: internal solver modules split by responsibility.
- `solver2d.py` / `solver3d.py`: compatibility wrappers over shared `solver_entrypoints.py`.

## Numerics contract

The current continuous-model and discrete-integrator contract is summarized in:

```text
docs/numerics_contract.md
```

The current numerics cleanup backlog and deferred behavior changes are tracked in:

```text
docs/numerics_remaining_tasks_20260405.md
```

## Included examples

- `examples/minimal_2d`
- `examples/minimal_3d`

Each example includes all files needed to run directly.

## Scratch outputs

- `_tmp_*` and `_out_*` directories are treated as scratch outputs.
- Timestamped comparison folders under `demo_output/reference_compare` are the durable benchmark artifacts.
- `solver_report.json` and `runtime_step_summary.csv` include `valid_mask` diagnostics plus `field_backend_kind`, and opt-in runs can also report `invalid_mask_stopped` particles.
- `tools/evaluate_valid_mask_rollout.py` is the standard way to compare `diagnostic` and `retry_then_stop` before considering default-on rollout.

## Current scope

This v29 line is intentionally focused on a **distributable, runnable package**.
It keeps the synthetic provider path as the stable packaged base. The next recommended step is to reconnect COMSOL field / geometry providers to this stable base and keep the same `PreparedRuntime` entry point.
