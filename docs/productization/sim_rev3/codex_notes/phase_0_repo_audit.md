# Phase 0 Repo Audit

Date: 2026-05-24
Workspace: `c:\Users\user\Desktop\partcle_sim-sim_rev2`

## Current State

Packaging note: this is the original Phase 0 snapshot. The planning docset
identified below was later archived under
`docs/productization/sim_rev3/archive/planning_docset_v2/`, and active
templates now live under `docs/productization/sim_rev3/templates/`.

- Current checkout reports branch `sim_rev4`.
- Local branches `sim_rev3` and `sim_rev4` both point at commit `5c9c99b4255f948fab752faa4c98cbfb3a89ec68`.
- The requested doc path `docs/productization/sim_rev3/` was not present before this audit file was created.
- The sim_rev3 planning docset is currently present as untracked content under `docs/sim_rev3_productization_docset_v2/`.
- The worktree already contains many modified, deleted, and untracked files unrelated to this audit. This phase does not revert or normalize them.

## Run Entry Points

Current main simulation run entry points:

- `run_from_yaml.py`: root wrapper around `particle_tracer_unified.cli:main`.
- `particle-tracer-run`: console script defined in `pyproject.toml`.
- `particle_tracer_unified/cli.py`: builds prepared runtime, writes prepare/check reports, and calls solver entry points.
- `particle_tracer_unified/solvers/solver_entrypoints.py`: dimension-checked solver helpers for 2D/3D and YAML-based runs.
- `particle_tracer_unified/io/runtime_builder.py`: central YAML/CSV/NPZ runtime construction and source preprocessing coordinator.
- `particle_tracer_unified/solvers/high_fidelity_runtime.py`: high-fidelity runtime loop.

Related utility entry points:

- `tools/build_comsol_case.py`: builds COMSOL-derived 2D solver cases from `.mphtxt` and field bundles.
- `tools/compare_against_reference.py`: runs a reference config and named candidates, then writes a comparison summary.
- `tools/collect_run_summaries.py`: aggregates existing run output directories into `run_summary_compare.csv`.
- `tools/export_visualizations.py` and related export scripts: visualization/report utilities for completed runs.

## COMSOL And VIGUS Inventory

COMSOL faithful/runtime integration:

- `particle_tracer_unified/io/comsol.py`: manifest-first faithful mode loader and guards.
- `particle_tracer_unified/io/comsol_manifest.py`: `comsol_case_manifest.yaml` parsing and strict validation.
- `particle_tracer_unified/io/comsol_release_reader.py`: COMSOL release table reader and particle-table conversion.
- `particle_tracer_unified/io/comsol_boundary_reader.py`: COMSOL boundary map and wall-law readers.
- `docs/comsol_parity.md`: current faithful comparison design notes.

Examples and data:

- `examples/comsol_faithful_2d/`: manifest-first COMSOL faithful template. It references generated files that must be supplied for a runnable case.
- `examples/comsol_from_data_2d/`: runnable COMSOL-derived data example with committed `generated/comsol_geometry_2d.npz` and `generated/comsol_field_2d.npz`.
- `data/argon_gec_ccp_base2.mphtxt` and `data/regridded_repo_field_bundle_argon_gec_ccp_base2_2d.npz`: small tracked inputs used by builder tests.
- `data/*.mph`: large tracked COMSOL model files, needing review for repository policy.

External COMSOL tooling:

- `external/comsol_particle_export/`: generic COMSOL particle export, raw export validation, field-bundle builders, release/table comparison, truth audit, and re-extraction promotion tools.
- `external/comsol_icp_export/`: ICP-specific external bridge and packer. It remains outside solver core and should not be imported by production solver code.

VIGUS-specific surfaces:

- No live `examples/vigus*` config was found.
- `_case_focus_ring_plasma_assumption_100/`, `_out_focus_ring_100_check/`, and `_out_focus_ring_100_run/` are current untracked case/output directories and should not be treated as production baselines.
- The audit found VIGUS lessons only in the productization docset, not as explicit solver-core wall ID logic.

## Compare Tools And Artifacts

Installed package compare CLIs from `pyproject.toml`:

- `particle-tracer-field-compare`: samples provider fields and writes `field_validation_error.csv`.
- `particle-tracer-acceleration-compare`: samples acceleration and writes `acceleration_error.csv`.
- `particle-tracer-trajectory-compare`: compares trajectory CSVs and writes `trajectory_error.csv`.
- `particle-tracer-boundary-compare`: compares first wall-hit diagnostics and writes `boundary_hit_comparison.csv`; optional summary JSON.

Additional compare/report tools:

- `tools/compare_against_reference.py`: writes timestamped `comparison_summary.json` under `demo_output/reference_compare` or a caller-provided output root.
- `tools/collect_run_summaries.py`: writes `run_summary_compare.csv`.
- `external/comsol_particle_export/compare_release_tables.py`: writes `matched_release_errors.csv` and `release_alignment_summary.json`.
- `external/comsol_particle_export/compare_particle_results.py`: writes `comparison_summary.json`, `comparison_by_state.csv`, `comparison_by_boundary.csv`, `matched_particle_errors.csv`, `force_model_alignment.json`, `release_alignment.json`, `trajectory_alignment.json`, `matched_trajectory_errors.csv`, `distribution_alignment.csv`, `field_alignment.json`, `trend_alignment.json`, and related readiness/divergence files.

Runtime output artifacts currently relevant to comparison:

- Always/core: `solver_report.json`, `final_particles.csv`, optional `prepared_runtime_summary.json`, `provider_contract_report.json`, `input_contract_report.json`.
- Source preprocessing: `source_model_summary.json`, `source_particle_diagnostics.csv`, `material_source_summary.csv`, `source_event_summary.csv`.
- Boundary/runtime diagnostics: `collision_diagnostics.json`, `wall_events.csv`, `wall_summary.json`, `wall_summary_by_part.csv`, `runtime_step_summary.csv`, `max_hit_events.csv`.
- Force/trajectory artifacts: `force_contributions.csv`, `positions_2d.npy` or `positions_3d.npy`, `save_frames.csv`, `segment_summary.csv`.

## Focused Test Coverage

`build_comsol_case` coverage:

- `tests/test_comsol_case_builder.py`: COMSOL precomputed case, geometry-only builder, default generated files, particles-only boundary release, field-bundle validation, boundary loop/geometry behavior, visualization helpers.
- `tests/test_comsol_icp_export.py`: ICP export field bundle, COMSOL edge/entity mapping, material inventory handling, wall catalog overrides, outward source velocity helper.

COMSOL faithful/readers coverage:

- `tests/test_comsol_manifest.py`: required faithful metadata, coordinate scale, velocity scale, force inventory.
- `tests/test_comsol_release_reader.py`: release table scaling, strict columns, non-finite values, duplicate particle IDs.
- `tests/test_comsol_boundary_reader.py`: boundary map and wall-law parsing, coverage, unsupported wall law details.
- `tests/test_comsol_faithful_runtime.py`: manifest release/wall loading, force inventory application, rejection of source preprocessing, non-strict masks, and field ghost cells.

`source_preprocess` / `boundary_release` coverage:

- `tests/test_runtime_builder_contracts.py`: boundary release offset, failed offset preflight errors, explicit boundary primitive requirement, far-outside particles, legacy alias rejection, COMSOL field source preprocessing.
- `tests/test_comsol_case_builder.py`: particles-only boundary release sources generated by the builder.

`high_fidelity_collision` / same-source skip coverage:

- `tests/test_boundary_runtime.py`: broad collision and wall-event behavior, physical hit time/velocity, repeated wall hits, contact sliding, boundary services, artifact modes, collision diagnostics.
- No explicit same-source skip test or obvious same-source skip string was found in the current test/code search. Treat this as a coverage gap for Phase 0.5, not as a failing behavior.

Sharded run aggregation coverage:

- No dedicated sharded runner or sharded aggregation test was found.
- Closest current coverage is `tools/collect_run_summaries.py` via `tests/test_run_summary_tools.py`, and reference/candidate summary aggregation in `tools/compare_against_reference.py` via `tests/test_runtime_builder_contracts.py`.
- Treat root artifact aggregation for sharded runs as a Phase 0.5/Phase 1 gap.

Comparison tools coverage:

- `tests/test_force_contribution_compare.py`: compare CLI help smoke and boundary first-hit summary output.
- `tests/test_comsol_particle_export.py`: release alignment, particle result comparison, field alignment, boundary-role derivation, export validation.
- `tests/test_run_summary_tools.py`: compact run summary aggregation.

## Test Results

Focused slice run:

```powershell
py -3 -m pytest -q tests/test_comsol_case_builder.py tests/test_comsol_manifest.py tests/test_comsol_release_reader.py tests/test_comsol_boundary_reader.py tests/test_comsol_faithful_runtime.py tests/test_runtime_builder_contracts.py tests/test_boundary_runtime.py tests/test_force_contribution_compare.py tests/test_run_summary_tools.py tests/test_comsol_particle_export.py tests/test_comsol_icp_export.py
```

Result:

```text
155 passed in 86.23s
```

Full suite run:

```powershell
py -3 -m pytest -q
```

Result:

```text
271 passed in 91.98s
```

Failure classification:

- No current test failures were observed.
- No evidence of current test drift versus true solver regression was found during this audit.
- Current risk is coverage gap, especially explicit same-source skip semantics and sharded root artifact aggregation.

## Cleanup Inventory

Keep:

- `particle_tracer_unified/`: production package, including COMSOL readers, source preprocessing, solver entry points, compare modules, providers, and current solver modules.
- `tools/build_comsol_case.py`, `tools/compare_against_reference.py`, `tools/collect_run_summaries.py`, and focused visualization/report utilities.
- `external/comsol_particle_export/` and `external/comsol_icp_export/` as external-only tooling.
- `examples/minimal_2d/`, `examples/minimal_3d/`, `examples/comsol_from_data_2d/`, and `examples/comsol_faithful_2d/`.
- Focused tests listed above.
- `docs/comsol_parity.md`, `docs/architecture.md`, `docs/numerics_contract.md`.

Delete later:

- `_out_focus_ring_100_check/`
- `_out_focus_ring_100_run/`
- `__pycache__/`, `.pytest_cache/`, `tools/__pycache__/`, `tests/__pycache__/`
- Ignored generated roots such as `_out_*`, `_tmp_*`, `_external_exports/`, `demo_output/`, `report_assets/`, and `examples/**/run_output*`.

Quarantine under research/docs if retained:

- `_case_focus_ring_plasma_assumption_100/`
- `report.md`
- `particle_tracer_all_figures.zip`
- `particle_tracer_decision_deck_with_figures_v4_complete.pptx`
- Historical probe outputs and old validation images/reports currently deleted in the dirty worktree, if they need to remain accessible for reference.
- VIGUS/focus-ring local probes should remain diagnostic evidence only, not product baselines.

Needs review:

- Large tracked `data/*.mph` files.
- Placement of `docs/sim_rev3_productization_docset_v2/` versus canonical `docs/productization/sim_rev3/`.
- Deleted historical docs/plans already present in the dirty worktree.
- Untracked solver refactor modules: `field_runtime.py`, `force_runtime.py`, `output_buffers.py`, `particle_state.py`, `runtime_plan.py`, `runtime_setup.py`.
- `tests/regression_helpers.py`, currently untracked but used by current tests.
- Whether old `docs/assets/icp_validation/*` generated assets should stay deleted, move to research/docs, or be regenerated outside product baselines.

## Recommended Next Phases

1. Phase 0.5 - focused tests and baseline smoke stabilization.
2. Phase 1 - minimal comparison rails.
3. Phase 2 - mode separation and minimal manifest gate.
4. Phase 3 - import, coordinate, and axisymmetric minimum semantics.
5. Phase 4 - release canonicalization.
6. Phase 5 - force breakdown and first-step parity.
7. Phase 6 - release grace and wall event simplification.
8. Phase 7 - axisymmetric RZ completion.
9. Phase 8 - V&V productization.
10. Phase 9 - cleanup, simplification, and performance.

Exact next phase: Phase 0.5 - focused tests and baseline smoke stabilization.

Reason: the focused suite and full suite are currently green, so the next useful work is to stabilize missing or weak focused checks before adding compare rails or changing physics. In particular, Phase 0.5 should cover explicit same-source skip behavior and sharded/root artifact aggregation enough that future phases can distinguish local probe improvements from global solver regressions.
