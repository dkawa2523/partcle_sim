# Post-Implementation Productization Gap Review

Date: 2026-05-25

Reviewed checkout: `sim_rev4` workspace after the sim_rev3 productization
phases. This review is an audit only; it does not change solver behavior,
tests, diagnostics, or cleanup state.

Packaging note: a later release-packaging pass moved active templates to
`docs/productization/sim_rev3/templates/` and quarantined the imported planning
docset under `docs/productization/sim_rev3/archive/planning_docset_v2/`. The
gap list below remains a historical snapshot of the post-implementation audit.

## Current Test State

Current full suite:

```powershell
py -3 -m pytest -q tests
```

Result:

```text
305 passed in 96.58s
```

`python` is still not the reliable launcher in this Windows workspace because
it resolves to the Windows Store alias in prior runs. Use `py -3` for local
verification commands.

## 1. What Is Now Implemented?

- Normal YAML-driven runs through `run_from_yaml.py` and
  `particle-tracer-run`.
- Central runtime/config loading in `particle_tracer_unified/io/runtime_builder.py`.
- Explicit run modes:
  - `comsol_faithful`
  - `surface_release_production`
- COMSOL faithful manifest gate requiring coordinate metadata, release table,
  boundary/wall-law files, field mapping, and force inventory.
- Runtime preservation/reporting of `coordinate_system`, `spatial_dim`, and
  canonical axis names.
- `axisymmetric_rz` import/report semantics, radial-axis validation, RZ
  geometry reporting, and ring-area weight utility.
- Source release canonicalization for explicit production boundary release:
  capture tolerance, inward offset, projection distance, selected boundary, and
  selected part diagnostics.
- COMSOL faithful release preservation: source preprocessing and boundary
  release are rejected instead of repairing COMSOL release coordinates.
- First-step comparison CLI:
  `particle-tracer-first-step-compare`.
- Deterministic force contribution compare artifacts for small first-step cases.
- Opt-in release grace for same-source outward wall crossings, with compact
  counters and blocked-reason counts.
- V&V workflow documentation under `docs/productization/sim_rev3/vv/`.
- Stable compare summary naming in `tools/compare_against_reference.py`.
- Shard-root compact artifact collection in `tools/collect_run_summaries.py`.
- Compact non-faithful output default through `output.mode: standard`, with
  debug/full artifacts still opt-in and faithful mode forced debug-equivalent.
- Cleanup/ignore rules for known historical generated artifact names.

## 2. Which Original Project Goals Are Now Covered?

Covered goals:

- Keep solver core generic and free of VIGUS-specific wall IDs or source-part
  hacks.
- Separate COMSOL faithful comparison from production surface-release behavior.
- Preserve raw COMSOL release coordinates in faithful mode.
- Allow explicit production boundary-release preprocessing without making it
  implicit.
- Make coordinate semantics visible enough that `axisymmetric_rz` is not
  silently treated as `cartesian_xy`.
- Compare more than endpoint counts: import/preflight, preprocess provenance,
  first-step state, force contributions, wall events, and ensemble summaries
  are all represented.
- Keep diagnostics compact by default and deep artifacts opt-in.
- Keep large COMSOL/VIGUS workflows out of normal unit tests.
- Provide focused tests for mode gates, release preprocessing, release grace,
  RZ semantics, first-step comparison, V&V artifact naming, and output modes.

## 3. Which Original Goals Are Still Not Covered?

Not fully covered:

- A single canonical runnable `surface_release_production` quickstart/example
  is still missing.
- The canonical requested template path
  `docs/productization/sim_rev3/templates/acceptance_matrix.csv` is still
  absent; the active matrix lives under `docs/productization/sim_rev3/vv/`.
- The repository is still dirty/untracked. Several imported runtime modules and
  productization docs are untracked, so the candidate is not yet release-safe.
- Large tracked COMSOL `.mph` files still need repository policy review.
- Full COMSOL/VIGUS-scale validation remains an operator workflow with
  user-supplied data, not a committed automated validation case.
- `axisymmetric_rz` does not implement full cylindrical dynamics, `v_theta`, or
  automatic ring-weighted source sampling.
- `output.save_every` is parsed into `OutputPlan` but the runtime loop still
  uses `solver.save_every`; the output-specific setting is currently a no-op.
- Standard output suppresses deep files, but still computes geometry summaries
  in normal runs, which may be too much work for large production cases.
- Sharded comparison aggregation lists/collects compact artifacts, but does not
  synthesize an ensemble comparison summary from shards.

## 4. Can A Normal Production Surface-Release Case Be Run Without COMSOL?

Yes, with caveats.

The solver can run non-COMSOL YAML cases using synthetic or precomputed
providers, explicit particle tables, explicit boundary primitives, and
`source.preprocess.boundary_release: true`. COMSOL is not required for the
production surface-release path.

The caveat is operator usability: README and numerics docs explain the pieces,
but there is not yet a small canonical `surface_release_production` example that
shows check-input, preprocessing, run command, and expected standard artifacts
end to end.

## 5. Can A COMSOL Faithful Comparison Case Be Run Without Hidden Solver-Side Repair?

Yes.

`comsol_faithful` is manifest-first and rejects solver-side source
preprocessing, implicit boundary release, field ghost cells, missing coordinate
scale, missing force inventory, and unsupported wall-law metadata. COMSOL
release tables remain the first-class release input. The solver does not read
`.mph` files and does not call the COMSOL API inside `particle_tracer_unified`.

The remaining limitation is data availability: the faithful example is a
template, so real parity runs still require exported COMSOL CSV/NPZ artifacts
and a complete manifest supplied by the operator or external tooling.

## 6. Are Import, Preprocess, First-Step, Wall-Event, And Ensemble Comparison Surfaces Available?

Yes.

- Import: input/provider preflight reports and COMSOL manifest validation.
- Preprocess: source model summary and source particle diagnostics when
  debug/explicit diagnostics are requested.
- First step: `particle-tracer-first-step-compare` with
  `first_step_error.csv`, `force_contributions.csv`, and compact JSON summary.
- Wall events: `particle-tracer-boundary-compare`, `wall_events.csv` in debug
  mode, wall summaries in standard mode, and collision diagnostics when
  requested.
- Ensemble: `tools/compare_against_reference.py` with timestamped and stable
  `comparison_summary.json`.
- Shards: `tools/collect_run_summaries.py --root-artifacts-dir` writes root
  summaries and a shard artifact manifest.

## 7. Are VIGUS-Specific Assumptions Absent From Solver Core?

Yes.

Current code search did not show VIGUS-specific branches, hard-coded VIGUS wall
IDs, or case-specific part-ID policies in `particle_tracer_unified`. ICP/VIGUS
material appears in docs, V&V wording, historical audits, data names, and
external tooling, not as solver-core behavior.

## 8. Are Diagnostics Minimal By Default?

Partially.

Artifact output is minimal enough by default: ordinary non-faithful runs use
`output.mode: standard`, which avoids trajectory arrays, wall-event CSVs,
runtime step summaries, source particle diagnostics, collision diagnostics, and
force contribution files unless explicitly requested.

However, standard-mode report generation still computes geometry summaries for
final/source/invalid-stop states. That is compact on disk but not necessarily
minimal in runtime cost for large production cases.

## 9. Are Old Exploratory Branches, _tmp Outputs, And Rejected Policies Isolated?

Partially.

Known focus-ring local outputs, root report/deck artifacts, and old ICP
validation assets were removed or ignored by exact/narrow names. Old policies
such as implicit faithful repair, COMSOL preprocessing, broad schema frameworks,
and VIGUS-specific solver branches are not in the production path.

Still unresolved:

- dirty/untracked worktree state;
- untracked productization docs and new runtime modules;
- large tracked COMSOL `.mph` files;
- legacy planning docset placement under `docs/sim_rev3_productization_docset_v2/`.

## 10. Next Highest-Value Improvements

1. Make the release candidate stageable and reproducible.
2. Add a concise production surface-release quickstart/example.
3. Fix output policy friction: `output.save_every` no-op and standard-mode
   geometry-summary cost.
4. Normalize productization docs/templates into canonical paths.
5. Decide large COMSOL asset policy.
6. Add one small sampled COMSOL/reference smoke workflow if suitable exported
   data can be curated without large fixtures.
7. Defer optional physics extensions until the product surface is stable.

## Remaining Item Classification

| Item | Classification | Reason |
| --- | --- | --- |
| Untracked imported runtime modules and test helpers | correctness blocker | The workspace passes tests, but a release/PR that misses these files would import-fail. Stage/review them before release. |
| No known failing solver physics behavior in current tests | correctness blocker | No active solver correctness blocker observed; full suite is green. Keep this row as current status, not proof of exhaustive validation. |
| Complete exported COMSOL faithful smoke case not committed | COMSOL parity blocker | Faithful machinery exists, but real parity still depends on operator-supplied exported CSV/NPZ/manifest artifacts. |
| Full COMSOL/VIGUS-scale validation not automated | COMSOL parity blocker | This is intentionally outside unit tests, but production readiness still needs an operator validation record. |
| Missing canonical `surface_release_production` quickstart/example | product usability blocker | Normal production can run without COMSOL, but a new engineer lacks one short, runnable product path. |
| Missing `docs/productization/sim_rev3/templates/acceptance_matrix.csv` | product usability blocker | The active matrix exists under `vv/`, but the requested canonical template path is absent. |
| `output.save_every` parsed but ignored | product usability blocker | Users can configure it without effect; either wire it into the loop or remove/document it. |
| Standard output still computes geometry summaries | performance blocker | Files are compact, but normal runs still pay geometry-summary sampling cost. |
| No large-case performance smoke baseline | performance blocker | The full test suite is green, but product-scale runtime/output overhead is not measured. |
| Dirty/deleted/untracked repository state | cleanup only | Needs staging/review hygiene before release; not a solver behavior change. |
| Large tracked `data/*.mph` files | cleanup only | Requires repository policy decision: keep as curated references, move, or replace with smaller exported fixtures. |
| Legacy docset under `docs/sim_rev3_productization_docset_v2/` | cleanup only | Useful history, but canonical placement remains unsettled. |
| Historical deleted reports/plans/assets | cleanup only | Phase 9 marks them non-production; final PR should intentionally keep/delete/quarantine them. |
| Full `v_theta` and cylindrical dynamics | optional extension | RZ semantics are explicit, but full axisymmetric physics was intentionally deferred. |
| Automatic ring-weighted source sampling | optional extension | Ring weighting utility exists for reporting/future use; sampling behavior is not changed automatically. |
| Shard ensemble summary merge | optional extension | Root artifact collection exists; automatic statistical ensemble merge can be added later if needed. |
| Dashboards or heavy V&V reporting | optional extension | Current productization intentionally favors compact artifacts over dashboard infrastructure. |

## Bottom Line

The implementation now covers the core sim_rev3 productization goals: explicit
modes, faithful COMSOL gates, production release preprocessing, coordinate/RZ
visibility, first-step/wall/ensemble compare surfaces, release grace, compact
artifacts, and focused tests.

It is not yet production-ready as a repository release because packaging,
canonical docs, surface-release onboarding, large-asset policy, and two output
policy/performance issues remain. The highest-value next move is a small
release-hardening pass, not new physics.
