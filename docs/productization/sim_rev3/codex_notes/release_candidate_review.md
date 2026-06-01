# sim_rev3 Productization Release Candidate Review

Date: 2026-05-24

Reviewed checkout: `sim_rev4` at `5c9c99b`. Phase 0 noted that `sim_rev3`
and `sim_rev4` pointed at the same commit at the start of productization work.

## Recommendation

Ready except listed blockers.

Packaging note: a later release-packaging pass moved the active templates to
`docs/productization/sim_rev3/templates/` and quarantined the imported planning
docset under `docs/productization/sim_rev3/archive/planning_docset_v2/`. The
review below is retained as the historical RC snapshot that identified those
gaps.

The solver behavior, mode gates, comparison tools, compact diagnostics, and
focused tests are in release-candidate shape. Before calling this production
ready, the repository still needs packaging cleanup: canonical productization
docs/templates must be placed consistently, the current dirty/untracked work
must be reviewed as a PR-sized change, and the production surface-release
operator path should have one small canonical runnable runbook or example.

## Reviewed Material

- `docs/productization/sim_rev3/codex_notes/phase_0_repo_audit.md`
- `docs/productization/sim_rev3/codex_notes/phase_9_cleanup_report.md`
- `docs/productization/sim_rev3/vv/README.md`
- `docs/productization/sim_rev3/vv/acceptance_matrix.csv`
- `docs/productization/sim_rev3/vv/task_record_template.md`
- `README.md`, `docs/comsol_parity.md`, examples, compare tools, runtime output
  plumbing, and current test inventory.

Requested `docs/productization/sim_rev3/templates/acceptance_matrix.csv` is not
present in this checkout. The active acceptance matrix is currently under
`docs/productization/sim_rev3/vv/acceptance_matrix.csv`.

## RC Questions

### 1. Can a new engineer run a normal production surface-release case?

Partially passed.

`README.md` documents normal runs through `run_from_yaml.py` and
`particle-tracer-run`, explains input checks, and names
`source.preprocess.boundary_release` as the production path for wall-origin
particles. `surface_release_production` mode allows explicit boundary-release
preprocessing while preserving standard non-faithful behavior.

Gap: there is not yet a single canonical productization runbook or runnable
`surface_release_production` example that walks an engineer through check-input,
preprocess, run, and expected compact outputs. The pieces exist, but the
operator path is still assembled from README plus examples.

### 2. Can a new engineer run a COMSOL faithful comparison?

Passed.

`docs/comsol_parity.md`, `README.md`, and `examples/comsol_faithful_2d` explain
the manifest-first COMSOL path. The expected inputs are explicit: coordinate
system and scale, release table, boundary map, wall laws, field mapping, and
force inventory. The docs also state that `.mph` access and COMSOL API/export
logic stay outside `particle_tracer_unified`.

The faithful example is intentionally a template until exported CSV/NPZ files
are supplied, which is appropriate for avoiding large fixtures in the package.

### 3. Are COMSOL faithful and production surface-release modes separated?

Passed.

`comsol_faithful` and `surface_release_production` are explicit modes. Faithful
mode rejects source preprocessing and boundary release, enforces strict manifest
metadata, keeps comparison artifacts available, and fails clearly on unknown
mode values. Production surface release remains opt-in through explicit
preprocess settings. Existing non-faithful runs keep normal defaults.

### 4. Are import, preprocess, first-step, wall-event, and ensemble comparisons available?

Passed.

Available comparison layers:

- Import/preflight: provider and input preflight reports from check-input/runtime
  preparation.
- Preprocess and release provenance: source preprocessing diagnostics when
  explicitly requested or in debug comparison workflows.
- First step: `particle-tracer-first-step-compare` writes first-step error,
  deterministic force contributions, and compact JSON summary.
- Wall events: boundary comparison CLI and debug wall-event artifacts.
- Ensemble: `tools/compare_against_reference.py` writes timestamped and stable
  root `comparison_summary.json`; `tools/collect_run_summaries.py` supports
  sharded root artifact collection.

The V&V guide separates synthetic verification, sampled COMSOL/reference
validation, and full reference validation.

### 5. Are VIGUS-specific hacks absent from solver core?

Passed.

Current code search found no production-path `VIGUS`/`vigus` branches, wall-ID
hacks, or case-specific source-part logic under `particle_tracer_unified`.
The remaining VIGUS references are documentation, historical audit context, or
operator-level validation language. Solver behavior is expressed through
coordinate systems, manifests, wall laws, source provenance, and explicit
configuration rather than VIGUS identifiers.

### 6. Are old exploratory branches or outputs quarantined or clearly not production?

Partially passed.

Phase 9 removed or ignored the exact local/generated roots identified in Phase
0, including focus-ring outputs, root report/deck artifacts, cache folders, and
old ICP validation assets. `.gitignore` now has narrow rules for recurring
historical artifact names.

Remaining deferred items are clearly documented but not fully resolved:

- large tracked `data/*.mph` / `.mphtxt` files require a repository policy
  decision;
- the legacy source docset remains under `docs/sim_rev3_productization_docset_v2/`;
- current changes are still dirty/untracked and need final review/commit
  hygiene before release.

### 7. Are diagnostics minimal by default?

Passed.

Ordinary non-faithful runs default to `output.mode: standard`, which keeps final
particles, solver report, wall/coating summaries, and compact preparation
reports. Deep artifacts such as trajectories, wall-event CSVs,
`runtime_step_summary.csv`, source particle diagnostics, collision diagnostics,
and force contributions require `output.mode: debug`, legacy
`output.artifact_mode: full`, or explicit `output.write_*` flags. Faithful
COMSOL comparison remains debug-equivalent by design.

### 8. Are tests focused and not excessive?

Passed.

The current test suite protects mode separation, coordinate semantics, release
canonicalization, first-step force comparison, release grace, RZ reporting, V&V
artifact naming, and compact output behavior with small synthetic or focused
fixtures. It does not require large VIGUS/COMSOL production cases in normal
unit tests.

Latest recorded Phase 9 run:

- `py -3 -m pytest -q tests`: `305 passed in 96.47s`
- focused output/V&V slice: `169 passed in 87.11s`

`python -m pytest -q tests` still hits the Windows Store alias in this
environment, so `py -3` is the reliable launcher here.

### 9. Are performance-sensitive paths free from unnecessary always-on logging?

Passed.

Default standard output avoids deep per-step/per-hit logging. Release grace uses
summary counters and compact reason counts rather than wall trace CSVs. RZ and
coordinate reporting is attached to summaries/preflight metadata rather than
hot-loop instrumentation. Debug and comparison artifacts are available only
when requested or required by faithful mode.

### 10. What remains before calling this production-ready?

Required before production-ready:

1. Move or copy the active acceptance matrix/template to the canonical requested
   path, or update the productization docs so there is only one authoritative
   location.
2. Add one concise production surface-release quickstart or runnable example
   using `surface_release_production`, explicit `boundary_release`, check-input,
   and expected standard artifacts.
3. Convert the current dirty/untracked repository state into a reviewed commit
   or PR, with obsolete deletions and new productization files intentionally
   staged.
4. Decide repository policy for large tracked COMSOL `.mph` assets.
5. Re-run the full suite after final packaging cleanup.

## Acceptance Matrix

| Area | Status | Reason |
| --- | --- | --- |
| Normal production surface-release workflow | partially passed | Core behavior and README guidance exist, but no single canonical runnable productization quickstart/example exists yet. |
| COMSOL faithful comparison workflow | passed | Manifest-first docs, example template, strict gates, and compare CLIs are present. |
| Faithful/production mode separation | passed | Faithful rejects preprocessing and requires manifest metadata; production boundary release is explicit. |
| Import/preprocess/first-step/wall/ensemble comparison availability | passed | Preflight reports, first-step compare, boundary compare, reference compare, and shard aggregation are available. |
| VIGUS-specific hacks absent from solver core | passed | Code search shows no VIGUS-specific solver branches or wall-ID logic. |
| Old exploratory outputs quarantined or excluded | partially passed | Local/generated artifacts were removed or ignored; large `.mph` files and docset placement remain deferred. |
| Minimal default diagnostics | passed | Standard output suppresses deep artifacts unless debug/explicit flags or faithful mode require them. |
| Focused tests, no excessive fixtures | passed | Full suite is green with synthetic/focused tests and no large COMSOL/VIGUS unit fixtures. |
| Hot-path logging/performance posture | passed | No always-on trace logging was added; deep diagnostics remain opt-in. |
| Canonical productization packaging | deferred with reason | `docs/productization/sim_rev3/templates/acceptance_matrix.csv` is missing and several release docs/files are still untracked. |
| Production-ready declaration | deferred with reason | Behavior is RC-ready, but documentation placement, surface-release quickstart, dirty worktree, and large-asset policy must be resolved first. |
