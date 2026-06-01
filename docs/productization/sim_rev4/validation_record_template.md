# sim_rev4 Validation Record Template

Use this template for release-candidate validation records. Commit the completed
Markdown record and any small summary files only. Do not commit large COMSOL
exports, full trajectory outputs, GIFs, `_out*` roots, or VIGUS-scale generated
artifacts.

Validation order: import -> preprocess -> first-step -> wall events -> ensemble.
Do not claim COMSOL parity or production readiness without evidence in every
required section below.

## Command Checklist

Run these from the repository root and paste the relevant outputs or artifact
paths into the sections below.

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
py -3 --version
py -3 -m pip show particle-tracer-unified
```

Record input asset hashes where feasible:

```powershell
Get-FileHash <case>/run_config.yaml -Algorithm SHA256
Get-FileHash <case>/<input>.csv -Algorithm SHA256
Get-FileHash <case>/<input>.npz -Algorithm SHA256
Get-FileHash <reference_or_comsol_export>.csv -Algorithm SHA256
```

Run the case preflight:

```powershell
py -3 run_from_yaml.py <case>/run_config.yaml --check-input --output-dir <record_root>/preflight
```

Run deterministic first-step and force contribution comparison:

```powershell
particle-tracer-first-step-compare --config <case>/run_config.yaml --reference <first_step_reference.csv> --output-dir <record_root>/first_step --stochastic off --seed <seed>
```

Run the candidate case:

```powershell
py -3 run_from_yaml.py <case>/run_config.yaml --output-dir <record_root>/solver_run
```

Run wall/boundary comparison when reference wall events are available:

```powershell
particle-tracer-boundary-compare --python <record_root>/solver_run/wall_events.csv --comsol <reference_wall_events.csv> --output <record_root>/boundary/boundary_hit_comparison.csv --summary <record_root>/boundary/boundary_hit_comparison.json
```

Run ensemble comparison:

```powershell
particle-tracer-compare-reference --reference-config <reference_run_config.yaml> --run candidate=<case>/run_config.yaml --output-root <record_root>/ensemble --reference-scope sampled --artifact-mode minimal
```

Validate compact artifacts:

```powershell
particle-tracer-validate-artifacts <record_root>/solver_run --workflow sampled
```

Run full COMSOL/reference diagnostics when COMSOL trajectory exports exist:

```powershell
particle-tracer-comsol-full-diagnostics --solver-output-dir <record_root>/solver_run --comsol-trajectory-csv <comsol_long_trajectory.csv> --comsol-release-csv <comsol_release_or_sample.csv> --first-step-dir <record_root>/first_step --reference-scope sampled --output-dir <record_root>/comsol_diagnostics
```

Use `--reference-scope full` only when the supplied COMSOL/reference export is a
full reference, not a sampled smoke subset.

## 1. Repository

- Validation date:
- Release candidate name/tag:
- Branch:
- Commit SHA:
- Dirty worktree at validation time: yes/no
- `git status --short` summary:
- Reviewer/operator:

## 2. Environment

- OS and version:
- Shell:
- Command launcher used: `py -3` / `python` / console scripts
- Python version:
- Installed package version:
- Important dependency versions:
- CPU/GPU/runtime notes:
- Timezone:

## 3. Input Assets

List every case, source, manifest, field, geometry, and reference artifact used.
For large external files, record the external location and checksum; do not copy
the file into the repository.

| Asset | Role | Path or external URI | Scope | SHA256 | Notes |
| --- | --- | --- | --- | --- | --- |
| run_config.yaml | candidate case config |  |  |  |  |
| particles/source table | candidate input |  |  |  |  |
| geometry bundle | candidate input |  |  |  |  |
| field bundle | candidate input |  |  |  |  |
| COMSOL manifest | faithful reference metadata |  |  |  |  |
| COMSOL release CSV | sampled/full reference |  |  |  |  |
| COMSOL trajectory CSV | sampled/full reference |  |  |  |  |

## 4. Random Seeds And Determinism

- Solver seed:
- Source preprocessing seed:
- First-step compare seed:
- Stochastic policy: `off` / `from-config`
- Brownian/Langevin enabled in production run: yes/no
- Deterministic rerun performed: yes/no
- Rerun artifact path:
- Notes:

## 5. Case Configuration

- Case config path:
- Mode: `comsol_faithful` / `surface_release_production` / other
- Coordinate system:
- Coordinate scale:
- Field backend:
- Drag model:
- Enabled force inventory:
- Wall law table path:
- Boundary map path:
- Source preprocessing settings:
- Output mode and explicit write flags:

## 6. Preflight And Import Results

Artifacts:

- `prepared_runtime_summary.json`:
- `provider_contract_report.json`:
- `input_contract_report.json`:
- `source_model_summary.json`:
- `source_particle_diagnostics.csv`:

Required checks:

- Coordinate system preserved and reported:
- Field/geometry axis alignment:
- Field support status: clean/mixed/hard_invalid counts
- Boundary/wall law coverage:
- Source provenance groups: known-source/unknown-source/production-generated
- Boundary release classification count:
- Boundary inward offset rule and value:
- Preflight verdict: pass/fail/deferred

## 7. First-Step Comparison

Artifacts:

- `first_step_compare_summary.json`:
- `first_step_error.csv`:
- optional dt sweep summaries:

Required metrics:

- Compared particle count:
- Post-preprocess state checked: yes/no
- Post-first-step state checked: yes/no
- Max position error:
- Mean position error:
- Max velocity error:
- Mean velocity error:
- Speed ratio summary:
- Field status counts:
- Acceptance threshold:
- Verdict: pass/fail/deferred

## 8. Force Contribution Comparison

Artifacts:

- `force_contributions.csv`:
- force contribution schema/version:

Required metrics:

- Force names present:
- Drag contribution checked:
- Electric contribution checked:
- Thermophoretic/DEP/lift/pressure-gradient/virtual-mass checks, if enabled:
- Brownian/stochastic handling:
- Total acceleration consistency:
- Missing force mappings:
- Verdict: pass/fail/deferred

## 9. Boundary And Wall Event Comparison

Artifacts:

- `wall_summary_by_part.csv`:
- `wall_events.csv` if debug/requested:
- `boundary_hit_comparison.csv`:
- `boundary_hit_comparison.json`:
- `collision_diagnostics.json` if debug/requested:

Required metrics:

- First-hit compared particle count:
- Wall part/action match summary:
- Solver-only event count:
- Reference-only event count:
- Unresolved crossing count:
- Max hits reached count:
- Numerical boundary stopped count:
- Release grace skip count:
- Release grace blocked count and reasons:
- Unsupported wall laws observed:
- Verdict: pass/fail/deferred

## 10. Ensemble Comparison

Artifacts:

- stable root `comparison_summary.json`:
- timestamped `compare_*/comparison_summary.json`:
- `run_summary_compare.csv`:
- `artifact_validation_summary.json`:
- sharded `shard_artifacts_manifest.json`, if applicable:

Required metrics:

- Reference scope: `sampled` / `full` / `unspecified`
- Candidate particle count:
- Reference particle count:
- Matched particle count:
- Class-match ratio:
- Final state fractions:
- Geometry feature deltas:
- Near-wall active no-hit summary:
- First-crossing/vacuum-time summary, if available:
- Runtime/collision counters:
- Endpoint counts used as sole acceptance criterion: must be `no`
- Verdict: pass/fail/deferred

## 11. Full COMSOL Diagnostics

Complete this section when COMSOL/reference trajectory exports are supplied.
Leave it explicitly deferred for synthetic-only or COMSOL-free production smoke
records.

Artifacts:

- `full_comsol_diagnostics_summary.json`:
- `suspicious_particles.csv`, if written:

Required metrics:

- COMSOL full particle count:
- COMSOL sampled particle count:
- Solver final particle count:
- Final vacuum fractions:
- Preprocess ratio metrics:
- First-step ratio metrics:
- Near-wall active counts:
- Zero-wall-hit fraction:
- Solver-only event count:
- COMSOL-only event count:
- Top source parts for residuals:
- Runtime and collision counters:
- Sampled/full distinction recorded: yes/no
- Verdict: pass/fail/deferred

## 12. Runtime And Performance Summary

- Candidate run output directory:
- Wall-clock run time:
- Particle count:
- Step count:
- Spatial dimension:
- Output mode:
- Trajectory artifacts written: yes/no
- Peak memory, if measured:
- Runtime counters:
- Collision counters:
- Sharded run: yes/no
- Shard count and aggregation artifact:
- Performance verdict: pass/fail/deferred

## 13. Known Deviations

List every accepted deviation from COMSOL/reference or expected production
behavior. Include whether it is a correctness blocker, COMSOL parity blocker,
product usability blocker, performance blocker, cleanup only, or optional
extension.

| ID | Classification | Description | Evidence | Owner | Release decision |
| --- | --- | --- | --- | --- | --- |

## 14. Final Verdict

Choose exactly one:

- `ready`
- `ready except listed blockers`
- `not ready`

Final verdict:

Required evidence summary:

- Preflight/import evidence complete: yes/no
- First-step and force evidence complete: yes/no
- Boundary/wall evidence complete: yes/no
- Ensemble evidence complete: yes/no
- Sampled/full COMSOL scope stated: yes/no
- Runtime/performance evidence complete: yes/no
- Known deviations reviewed: yes/no

Release sign-off:

- Operator:
- Reviewer:
- Date:

## Production Release Evidence Requirement

Before production release, the validation record must include:

- exact branch and commit SHA for the candidate;
- reproducible environment and command launcher;
- case config, input asset list, and hashes or external artifact references;
- random seeds and stochastic policy;
- passing preflight/import checks;
- first-step error and force contribution artifacts;
- boundary/wall event comparison or an explicit reason it is not applicable;
- ensemble comparison with sampled/full reference scope clearly stated;
- runtime/performance summary from the validated candidate;
- known deviations with release decisions;
- a final verdict signed by the operator/reviewer.

Endpoint counts alone are not sufficient evidence for production release.
