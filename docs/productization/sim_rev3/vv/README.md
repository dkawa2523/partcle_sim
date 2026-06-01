# Phase 8 V&V Workflow

This workflow keeps verification, sampled validation, and full-reference validation separate. It is intended to produce compact, reproducible artifacts without requiring large COMSOL or VIGUS data in normal tests.

Product comparison order: import -> preprocess -> first-step -> wall events -> ensemble.

The acceptance matrix and task record template live in the canonical template
directory:

- `docs/productization/sim_rev3/templates/acceptance_matrix.csv`
- `docs/productization/sim_rev3/templates/task_record_template.md`

## Verification

Verification cases use synthetic inputs and deterministic settings. They check solver mechanics against known behavior before any COMSOL/reference comparison.

Run the quick synthetic gate:

```powershell
py -3 -m pytest -q tests/test_force_contribution_compare.py tests/test_boundary_runtime.py tests/test_runtime_builder_contracts.py tests/test_run_summary_tools.py
```

Primary checks:

- Import and provider contracts preserve coordinate system, axes, and field support.
- Preprocess reports release classification and projection/offset provenance when enabled.
- First-step comparison writes `first_step_error.csv`, `force_contributions.csv`, and `first_step_compare_summary.json`.
- Wall-event tests cover first-hit handling and release-grace behavior without endpoint-count tuning.
- Ensemble summaries use state fractions, boundary/event diagnostics, and geometry features, not only final endpoint counts.

## Sampled Validation

Sampled validation compares short or reduced reference cases. Use it for COMSOL-like or exported-reference smoke checks that fit in local development.

Recommended sequence:

```powershell
particle-tracer-build-comsol-case --help
py -3 -m particle_tracer_unified.compare.first_step_compare --config <run_config.yaml> --output-dir <root>/first_step
py -3 -m particle_tracer_unified.compare.boundary_compare --python <wall_events.csv> --comsol <reference_wall_events.csv> --output <root>/boundary_hit_comparison.csv --summary <root>/boundary_hit_comparison.json
particle-tracer-compare-reference --reference-config <reference.yaml> --run candidate=<candidate.yaml> --output-root <root>/ensemble --reference-scope sampled --artifact-mode minimal
particle-tracer-validate-artifacts <solver_or_comparison_root> --workflow sampled
```

Expected compact artifacts:

- `first_step_error.csv`
- `force_contributions.csv`
- `boundary_hit_comparison.csv`
- `boundary_hit_comparison.json`
- timestamped `compare_*/comparison_summary.json`
- stable root `comparison_summary.json`
- `artifact_validation_summary.json` when artifact validation is run

Normal solver runs default to compact `output.mode: standard`. Use
`output.mode: debug` or explicit `output.write_*` flags for workflows that need
solver-generated `wall_events.csv`, `runtime_step_summary.csv`,
`source_particle_diagnostics.csv`, `collision_diagnostics.json`, or trajectory
arrays.

## Visualization Artifacts

Graph export is a report aid and can run from compact standard output:

```powershell
particle-tracer-export-visualizations --output-dir <solver_out> --modules graphs
```

Large GIF generation is not a validation gate. For COMSOL comparison reports,
prefer a sampled side-by-side comparison GIF when sampled solver and reference
trajectories are available. Full solver GIFs should be generated only from
debug or explicit trajectory-saving runs, with particle/frame limits:

```powershell
particle-tracer-export-visualizations --output-dir <solver_out> --case-dir <case_dir> --modules animations --animation-max-particles 1000 --animation-max-frames 180 --skip-all-particles-animation
```

If animation fails or trajectory files are absent, use compare summaries,
first-step artifacts, wall summaries, and graph exports as the official
validation evidence. See `docs/visualization_workflow.md`.

## Full Reference Validation

Full COMSOL/VIGUS-scale validation is an operator workflow. Keep large input data and generated outputs outside unit tests and outside commits unless explicitly curated as a fixture.

Recommended command shape:

```powershell
particle-tracer-compare-reference --reference-config <full_reference.yaml> --run production=<candidate.yaml> --output-root <root>/full_reference --reference-scope full --per-run-timeout-s 0 --artifact-mode minimal
particle-tracer-validate-artifacts <solver_or_comparison_root> --workflow full --require-source-diagnostics
```

Acceptance should inspect import, preprocess, first-step, wall events, ensemble distributions, and runtime diagnostics. Do not accept on endpoint counts alone.

`validate_comparison_artifacts.py` validates a root directory that contains the
compact solver artifacts plus the stable `comparison_summary.json`. For direct
`compare_against_reference.py` output roots, solver artifacts live in the
timestamped comparison subdirectories; validate the solver/sharded root that
will feed residual or full diagnostics.

## Sharded Runs

After sharded runs finish, collect compact root artifacts:

```powershell
particle-tracer-collect-summaries <shard_output_1> <shard_output_2> --root-artifacts-dir <root>
```

This writes:

- `<root>/run_summary_compare.csv`
- `<root>/solver_report.json`
- `<root>/prepared_runtime_summary.json`
- `<root>/wall_summary_by_part.csv`
- `<root>/source_model_summary.json` when shard source summaries are present
- `<root>/source_particle_diagnostics.csv` when shard diagnostics are present
- `<root>/first_step_compare_summary.json` when shard first-step summaries are present
- `<root>/collision_diagnostics.json` only when shard debug diagnostics are present
- `<root>/shard_artifacts_manifest.json`

Per-shard comparison summaries are listed in the manifest but are not merged. Ensemble comparison should produce its own root `comparison_summary.json` with a compare tool.

Validate a sharded root before running root-level diagnostics:

```powershell
particle-tracer-validate-artifacts <root> --workflow sharded --require-source-diagnostics
```

Add `--require-first-step` or `--require-debug` only when those artifacts are
part of the intended review.

## Residual Gap Snapshot

When an operator has current run and comparison artifacts, write a compact gap
summary without rerunning or tuning the solver:

```powershell
particle-tracer-residual-gap-summary --run-output-dir <out_run> --preflight-dir <out_check> --first-step-dir <out_first_step> --boundary-summary <boundary_hit_comparison.json> --ensemble-summary <comparison_summary.json> --reference-scope sampled --output-dir <gap_out>
```

The tool writes:

- `<gap_out>/current_residual_gap_summary.json`
- `<gap_out>/current_residual_gap_report.md`

Missing optional artifacts are reported in the summary. Use
`--reference-scope full` only for full COMSOL/reference data; do not use
historical `_tmp*` output roots as official baselines.

For a reusable full COMSOL/reference diagnostics pass over one existing solver
output and one COMSOL long trajectory export:

```powershell
py -3 -m particle_tracer_unified.compare.comsol_full_diagnostics --solver-output-dir <out_run> --comsol-trajectory-csv <comsol_long_trajectory.csv> --comsol-release-csv <comsol_release_or_sample.csv> --first-step-dir <out_first_step> --reference-scope sampled --output-dir <diag_out>
```

The console entry point is equivalent:

```powershell
particle-tracer-comsol-full-diagnostics --solver-output-dir <out_run> --comsol-trajectory-csv <comsol_long_trajectory.csv> --comsol-release-csv <comsol_release_or_sample.csv> --reference-scope full --output-dir <diag_out>
```

This writes `<diag_out>/full_comsol_diagnostics_summary.json` and, when
residuals are found, a compact `<diag_out>/suspicious_particles.csv`. The
summary keeps sampled and full reference counts separate and reports final
snapshot metrics separately from wall-event/trajectory proxies.
