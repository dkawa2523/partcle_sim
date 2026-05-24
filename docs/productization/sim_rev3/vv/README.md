# Phase 8 V&V Workflow

This workflow keeps verification, sampled validation, and full-reference validation separate. It is intended to produce compact, reproducible artifacts without requiring large COMSOL or VIGUS data in normal tests.

Product comparison order: import -> preprocess -> first-step -> wall events -> ensemble.

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
py -3 tools/build_comsol_case.py --help
py -3 -m particle_tracer_unified.compare.first_step_compare --config <run_config.yaml> --output-dir <root>/first_step
py -3 -m particle_tracer_unified.compare.boundary_compare --python <wall_events.csv> --comsol <reference_wall_events.csv> --output <root>/boundary_hit_comparison.csv --summary <root>/boundary_hit_comparison.json
py -3 tools/compare_against_reference.py --reference-config <reference.yaml> --run candidate=<candidate.yaml> --output-root <root>/ensemble --reference-scope sampled --artifact-mode minimal
```

Expected compact artifacts:

- `first_step_error.csv`
- `force_contributions.csv`
- `boundary_hit_comparison.csv`
- `boundary_hit_comparison.json`
- timestamped `compare_*/comparison_summary.json`
- stable root `comparison_summary.json`

Normal solver runs default to compact `output.mode: standard`. Use
`output.mode: debug` or explicit `output.write_*` flags for workflows that need
solver-generated `wall_events.csv`, `runtime_step_summary.csv`,
`source_particle_diagnostics.csv`, `collision_diagnostics.json`, or trajectory
arrays.

## Full Reference Validation

Full COMSOL/VIGUS-scale validation is an operator workflow. Keep large input data and generated outputs outside unit tests and outside commits unless explicitly curated as a fixture.

Recommended command shape:

```powershell
py -3 tools/compare_against_reference.py --reference-config <full_reference.yaml> --run production=<candidate.yaml> --output-root <root>/full_reference --reference-scope full --per-run-timeout-s 0 --artifact-mode minimal
```

Acceptance should inspect import, preprocess, first-step, wall events, ensemble distributions, and runtime diagnostics. Do not accept on endpoint counts alone.

## Sharded Runs

After sharded runs finish, collect compact root artifacts:

```powershell
py -3 tools/collect_run_summaries.py <shard_output_1> <shard_output_2> --root-artifacts-dir <root>
```

This writes:

- `<root>/run_summary_compare.csv`
- `<root>/source_particle_diagnostics.csv` when shard diagnostics are present
- `<root>/shard_artifacts_manifest.json`

Per-shard comparison summaries are listed in the manifest but are not merged. Ensemble comparison should produce its own root `comparison_summary.json` with a compare tool.
