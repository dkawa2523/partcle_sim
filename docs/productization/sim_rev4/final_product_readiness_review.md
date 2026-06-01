# sim_rev4 Final Product Readiness Review

Date: 2026-05-28
Workspace reviewed: `c:\Users\user\Desktop\partcle_sim-sim_rev2`

## Verdict

Ready except blockers.

The product surface is coherent enough for a release candidate: normal
surface-release production, COMSOL faithful comparison, non-COMSOL adapter
guidance, compact diagnostics, compare tools, packaging policy, and validation
record workflow are all present. It is not yet production-release ready because
release evidence is still missing: no completed validation record with target
COMSOL/reference data is committed, no large-case performance baseline is
recorded, and the current workspace still needs final staging/review hygiene.

## Blockers

| ID | Type | Blocker | Required Resolution |
| --- | --- | --- | --- |
| B1 | Validation evidence | No completed sim_rev4 validation record with real sampled/full reference artifacts is present. | Complete `docs/productization/sim_rev4/validation_record_template.md` for the release candidate and store only compact summary artifacts or external hashes. |
| B2 | Performance evidence | No product-scale performance measurement or bottleneck report is present. | Run and record a large-case smoke/performance baseline including particle count, step count, wall-clock time, output mode, runtime counters, and collision counters. |
| B3 | Release hygiene | The workspace contains many intended productization changes, moves, deletions, and untracked files. | Stage/review as a coherent release PR or commit set; verify no generated outputs or large new COMSOL assets are included. |
| B4 | Asset disposition | Large legacy `data/*.mph` files are governed by policy but still tracked pending migration decision. | Decide whether to keep as legacy tracked assets, move to external storage, or replace with exported CSV/NPZ fixtures before public/product distribution. |

## Readiness Questions

### 1. Can a new engineer run a normal production surface-release case?

Yes.

`README.md` points to `examples/minimal_surface_release_production/README.md`,
which gives check-input and run commands. The example is COMSOL-free, uses
`surface_release_production`, enables
`source.preprocess.boundary_release: true`, and documents expected standard
artifacts plus debug artifacts. This is now a clear operator path.

### 2. Can a new engineer run a COMSOL faithful comparison?

Yes, with exported COMSOL artifacts supplied by the operator.

`docs/comsol_onboarding.md`, `docs/comsol_parity.md`, and
`examples/comsol_faithful_2d/` explain the manifest-first workflow, required
release/boundary/wall/field/force inputs, and comparison order. The solver does
not read `.mph` files or use COMSOL APIs inside `particle_tracer_unified`.

### 3. Can a new engineer add a non-COMSOL input adapter without changing solver core?

Yes.

`docs/canonical_input_bundle.md` defines the adapter boundary: normalized
particle/source tables, optional material/wall/process tables, canonical
geometry provider, canonical field provider, coordinate system, units, wall law
coverage, source provenance, and runtime loading expectations. It explicitly
instructs future adapters to normalize into the canonical bundle and pass
objects through `runtime_builder.py` without source-specific solver branches.

### 4. Are import, preprocess, first-step, wall-event, ensemble, and runtime comparisons available?

Yes.

- Import/preflight: `run_from_yaml.py --check-input`, provider/input reports,
  manifest validation.
- Preprocess: source model summary and source particle diagnostics when
  requested or in comparison/debug workflows.
- First-step: `particle-tracer-first-step-compare`, `first_step_error.csv`,
  `force_contributions.csv`, and summary JSON.
- Wall events: `particle-tracer-boundary-compare`, wall summaries, optional
  debug `wall_events.csv`, and collision diagnostics.
- Ensemble: `particle-tracer-compare-reference`, stable/timestamped
  `comparison_summary.json`, and `run_summary_compare.csv`.
- Runtime/root artifacts: `particle-tracer-validate-artifacts`,
  `particle-tracer-collect-summaries`, and
  `particle-tracer-comsol-full-diagnostics`.

### 5. Are boundary capture and inward offset clearly separated?

Yes.

`docs/numerics_contract.md` states that
`source.preprocess.boundary_capture_tolerance_m` controls only near-boundary
classification, while `source.preprocess.boundary_inward_offset_m` controls the
small inward displacement after classification. The documented default keeps
inward offset tied to small epsilon/on-boundary tolerance and independent of an
explicitly large capture tolerance.

### 6. Are source provenance and source_id=0 semantics clear?

Yes.

`docs/numerics_contract.md` and `docs/canonical_input_bundle.md` define
`source_part_id > 0` as known source boundary provenance and
`source_part_id == 0` as unknown/absent provenance. Unknown source remains
unknown, is not repaired to nearest wall, and is not eligible for same-source
release grace. Production boundary release may report projected boundary data
without rewriting unknown input provenance.

### 7. Are COMSOL wall laws mapped or failed clearly?

Yes.

`docs/wall_law_catalog.md` lists supported, limited, and unsupported wall laws.
`comsol_faithful` requires an explicit wall law file and fails on unknown or
unsupported mappings. Production runs may use explicit supported laws, but
unknown wall laws fail rather than silently approximating.

### 8. Are diagnostics minimal by default?

Yes for artifacts.

Normal non-faithful runs default to compact standard output: final particles,
solver report, wall/coating summaries, and compact preparation reports. Deep
artifacts such as trajectory arrays, wall-event CSVs, step summaries, source
particle diagnostics, collision diagnostics, and force contributions require
debug/full mode or explicit output flags. COMSOL faithful mode remains
debug-equivalent where comparison artifacts are needed.

Residual risk: runtime cost of summary generation for product-scale cases is
not yet benchmarked in a recorded performance run.

### 9. Are large outputs and COMSOL assets governed by policy?

Yes, with a remaining disposition decision.

`docs/release_packaging_policy.md` defines canonical docs/templates, example
rules, COMSOL asset policy, generated-output exclusions, and console scripts.
`.gitignore` excludes generated roots and new COMSOL binary model artifacts.
Existing legacy `data/*.mph` assets are explicitly called out as retained until
a separate migration/externalization decision.

### 10. Are performance bottlenecks measured?

Not enough for production release.

The test suite exercises correctness and compact-output behavior, and default
diagnostics avoid always-on deep logging. However, this checkout does not
contain a product-scale performance record with wall-clock time, memory, shard
behavior, output mode comparison, or identified bottlenecks. This is blocker B2
for production release.

### 11. Is there at least one validation record workflow?

Yes.

`docs/productization/sim_rev4/validation_record_template.md` provides a
reproducible validation record workflow covering branch/commit, environment,
input hashes, seeds, case config, preflight, first-step, force contributions,
wall events, ensemble comparison, COMSOL sampled/full scope, runtime summary,
known deviations, and final verdict. It also states that endpoint counts alone
are not sufficient evidence.

### 12. What remains before production release?

Before production release:

1. Complete one validation record for the release candidate using the template.
2. Include sampled or full COMSOL/reference evidence, or explicitly scope the
   release to COMSOL-free production smoke only.
3. Record product-scale performance and runtime/collision counters.
4. Finalize large COMSOL asset disposition, especially legacy tracked `.mph`
   files.
5. Stage and review the full release change set, confirming no generated
   outputs or large new artifacts are included.
6. Re-run the full test suite from the staged state.

## Deferred Extensions

- Full COMSOL/VIGUS-scale automated validation in CI.
- Full cylindrical `v_theta` dynamics for `axisymmetric_rz`.
- Automatic ring-weighted source sampling.
- Field-driven ion drag and explicit sheath/near-wall correction interfaces.
- Larger performance benchmark harness and trend history.
- Dashboard-style V&V reporting; current workflow intentionally stays compact.

## Recommended Next Release Milestone

Recommended next milestone: `sim_rev4-validation-record`.

Scope it narrowly:

- complete one sampled COMSOL/reference validation record;
- complete one production surface-release smoke validation record;
- record a product-scale performance baseline;
- make the large-asset disposition decision;
- cut a clean release PR/commit set with no generated outputs.

Do not add new physics in that milestone unless validation exposes a blocker.

## Tests Run For This Review

Executed:

```powershell
git diff --check -- docs/productization/sim_rev4/final_product_readiness_review.md
rg -n "^## Verdict|^## Blockers|^### 1\.|^### 2\.|^### 3\.|^### 4\.|^### 5\.|^### 6\.|^### 7\.|^### 8\.|^### 9\.|^### 10\.|^### 11\.|^### 12\.|^## Deferred Extensions|^## Recommended Next Release Milestone|^## Tests Run" docs/productization/sim_rev4/final_product_readiness_review.md
py -3 -m pytest -q tests
```

Result:

```text
331 passed in 99.78s
```

Recommended release gate:

```powershell
py -3 -m pytest -q tests
particle-tracer-validate-artifacts <validation_root> --workflow sampled
particle-tracer-validate-artifacts <validation_root> --workflow full --require-source-diagnostics
```

Use `--workflow full` only when full reference artifacts are actually supplied.
