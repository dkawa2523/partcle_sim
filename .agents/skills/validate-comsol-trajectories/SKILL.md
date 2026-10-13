---
name: validate-comsol-trajectories
description: Evaluate particle trajectories and physical outcomes against COMSOL, from saved-result practical agreement to qualified matched comparisons, stochastic populations, and discrepancy attribution. Use for comparison criteria, diagnosis, or improvement assessment; inspection owns native evidence and preparation owns solver inputs and numerical settings.
---

# Evaluate the requested particle comparison

Produce a bounded answer to the physical question, with a claim no stronger
than its evidence. A brief instruction does not require a historical run matrix.
Examples of sufficient instructions:

- 「このMPHを本solverで再現し、wafer到達率と到達時刻を収束評価して。」
- 「この2つの保存結果の差を調べ、設定・場転送・本体のどこが原因か絞って。」

Infer the objective and reuse evidence-backed settings. For an unspecified
trajectory comparison, start with whole-population spatial views, time-resolved
transport differences when pairing is meaningful, and outcome/coverage summaries.
Add first arrival or residence only when relevant to the physical question;
a boundary-free transport problem needs no deposition metric. Label inferred
priorities provisional. Ask only for a missing
choice that materially changes the task, continuing independent work.
Existing chamber dimensions, tags, boundary IDs, seeds, steps, margins and
historical PASS records are never new-case defaults.

## Select the smallest mode that fulfills the request

| Mode | Minimum usable inputs | Output and completion/stop condition |
|---|---|---|
| **Comparison plan**: feasibility or experiment design | Objective and an MPH/export or inspection record identifying what is known | Compared/excluded physics, feasible intent, primary observables, missing evidence and bounded next experiment. Finish when the plan or precise blocker is known; no simulation is implied. |
| **Existing-data characterization**: saved-output comparison | Reference/candidate states, events or population tables; known units and identities for compared quantities | Raw SI differences, fate/arrival summaries, coverage/censoring and qualified/untested claims. Finish when supplied data answer the descriptive question or its limitations are explicit; new native solves/convergence are not prerequisites. |
| **Decisive pilot diagnostic**: unexplained mismatch or unfamiliar execution | Discrepancy, relevant inspected meaning and prepared inputs, or the specific missing evidence | One first-divergence packet, discriminating probes and an evidenced owner or exact missing observation. Stop on supported cause, missing prerequisite, error floor or hypothesis budget. |
| **Registered converged comparison**: accuracy-qualified agreement or completed reproduction | Actual compared model/field/source/boundary bindings, prepared cases, justified independent numerical plan, and qualified saved levels or runnable participants | Registered physical metrics, uncertainty and scoped verdicts. Finish when tests are decided, or deliver usable results with exact untested scope/blocker. Execute only levels needed for that claim. |

These are work scopes, not mandatory stages or new statuses. Carry an execution
and evaluation request through to its requested comparison; a pilot may resolve
prerequisites. A saved-result or planning request must not expand into native
solves. A local diagnostic can be complete while whole-trajectory parity is
`NOT_TESTED`.

## Judge the requested use, with separate claims

Read [agreement-and-observables.md](references/agreement-and-observables.md)
when selecting metrics or answering whether trajectories/results are close
enough. Report visual spatial similarity, practical observable agreement,
numerically qualified solver agreement, and physical validity separately, at
the depth requested. These are claim descriptions, not new tool statuses.
An analogous model can be useful for transport even though strict equation
parity is untested. Conversely a similar bundle can hide wrong arrival times
or particle outcomes. `NOT_TESTED` applies to the named unqualified claim,
not to every result or to the usefulness of a saved-data comparison.

Use the user's purpose and any supplied allowances. If none are available,
quantify observed differences and coverage and give a scoped qualitative
conclusion; choose a scientifically justified provisional allowance only when
it helps, label any use of already viewed results, and never retrofit a
confirmatory PASS. A user's visual acceptance applies to the present view;
it supplies no universal tolerance for another chamber.

## Establish evidence for this mode

Locate the root containing `particle_platform_redesign/AGENTS.md` and
`particle_platform_redesign/solver/pyproject.toml`. Read root/redesign instructions
and relevant sections of current authorities, rather than all historical cases:

- `particle_platform_redesign/vv_methodology.md`: comparison meaning and claims.
- `particle_platform_redesign/solver/docs/physics_models.md`: selected equations,
  primitives and applicability; inspect source when revisions conflict.
- `particle_platform_redesign/solver/docs/numerics.md` and
  `support_and_errors.md`: restrictions relevant to an executed method/model.
- `particle_platform_redesign/solver/tools/vv/comsol/README.md`: current preflight
  and the selected tool's scope.

Reuse [inspect-comsol-model](../inspect-comsol-model/SKILL.md)'s extraction record
and immutable native evidence. Reinspect only changed/unresolved bindings,
including external input/library dependencies described in its
[extraction record](../inspect-comsol-model/references/extraction-record.md).
Feature names are not equations; exports alone do not establish MPH access.
Missing native access does not prevent descriptive evaluation of known exports.

Bind actual MPH/solution/dataset/dependencies, candidate case/field/result,
executor/evaluator and conversion lineage. New runs also record lock/Python/
COMSOL versions, commands, completion and resolved revisions; `run.json` owns
completed-run revisions. A historical source roster does not identify an
installed executor.

Read [evaluation-and-attribution.md](references/evaluation-and-attribution.md)
for metrics, numerical qualification or diagnosis. Method, `dt_s`, spatial and
Brownian resolution selection have one owner:
[numerical-plan.md](../prepare-comsol-field-case/references/numerical-plan.md).
Consume that plan; never use COMSOL output times as candidate production steps
or keep halving dt through a field/model error floor.

## Make the compared scope explicit

| Intent | Required meaning and allowed interpretation |
|---|---|
| **Matched full case** | Same equations, primitive authority, realized source, contact and wall meaning over the requested case. Numerical agreement needs adequate independent resolutions. |
| **Matched slice** | Named particle/domain/time/physics scope with the same actual governing problem on both sides. Excluded active physics must be removed from both participants and the altered reference recomputed. An existing trajectory slice is valid only where omitted physics is demonstrably absent/inactive. Subtracting a force from saved states cannot create a matched trajectory. |
| **Common-model companion** | A source-preserving reference copy uses the supported common model. Read back and solve that reference; retain differences from the original MPH. Agreement does not certify the original. |
| **Original versus analogous model** | Explicit approximation, regime and question. Differences measure model-form sensitivity; unsupported meaning cannot receive solver-parity PASS/FAIL. Independently verify the candidate and report outcomes. |
| **Insufficiently comparable** | Report supported quantities, unknown meaning and the smallest missing export/capability. Preserve useful raw differences without claiming numerical parity. |

A similar feature label is not an equivalent equation. Missing primitives do
not justify selecting an easier model. Reduced DOF/contact and effective-gas/
property conventions belong to preparation's model crosswalk.

Use one compact comparison manifest, or the existing extraction/planning record
with linked sections, for objective, mode/intent, artifacts, crosswalk, metrics,
exclusions, uncertainty and investigation budget. Do not duplicate inventories,
particle tables or numerical plans into several new ledgers.

## Apply gates to the requested claim

For a new certified comparison use existing `semantic_inventory.json` v2 and
`meaning_preflight.py`, binding selected expected/actual observations by path,
SHA-256 and JSON pointer. Reuse valid artifacts and refresh bindings at
consumption. `ADAPTER_REQUIRED` needs conversion and actual post-conversion
readback; `AMBIGUOUS` leaves the affected claim untested. Hash/value equality
proves binding, not native authenticity, assembled RHS, accuracy or validity.

Keep the existing questions separate:

- `same_canonical_field_solver_parity`: actual participants execute the same
  canonical field identity and evaluation meaning. Filenames/node values alone
  do not prove this. Field production/import/recovery error is excluded.
- `native_fe_end_to_end_reproduction`: native COMSOL FE versus candidate with
  producer/adapter lineage. Field representation/recovery and geometry error
  are included in the result.

Run from `particle_platform_redesign/solver/`, with an unused output directory:

```console
uv run --locked python tools/vv/comsol/meaning_preflight.py path/to/semantic_inventory.json --output path/to/new_preflight_directory
```

Only a preflight parity/reproduction question whose condition is `SUPPORTED`
can receive a qualified `PASS` or `FAIL`; readiness is not a numerical verdict.
This gate does not govern descriptive/practical observable comparisons or
independent candidate verification. Existing-data characterization and local
probes may proceed without certifying unknown layers; state limitations and
use `NOT_TESTED` for qualified parity. Field-only preparation belongs to
prepare. With neither question requested, current preflight exit 2 means no
comparison readiness was requested, not that all preparation layers failed;
inspect reports instead of inventing a comparison to get exit zero.

## Execute the necessary experiment and decide

For inputs follow [prepare-comsol-field-case](../prepare-comsol-field-case/SKILL.md).
Preserve original MPH bytes. Re-solve a copy only for requested missing reference,
companion or convergence evidence; valid bound saved evidence can replace a new
solve. Start unfamiliar execution with representative particles/surfaces, then
enlarge only as needed for the physical/statistical claim. Candidate execution
uses `load_case -> simulate -> open_result` or public `check/run/inspect`, from
the independent uv project with `uv run --locked`. Inspect integrity, failures,
population coverage and revisions; `check` alone is no runtime/convergence proof.

Existing exporters/evaluators are case-bound until their source/CLI prove
otherwise. `diagnose_matched_layers.py`, for example, hard-codes `287 * 41` rows
and a narrow force profile. Adapt a small external tool to actual bindings when
necessary. Do not introduce a universal COMSOL framework, private production
orchestration, second engine or core COMSOL branch.

For supported deterministic pre-event states, use the current
`tools.vv.comsol.evaluate_matched_trajectory` through its module CLI and read
the generic descriptor contract in `tools/vv/comsol/README.md`. It separates
actual integration steps from explicit observation times, aligns actual IDs
and includes charge. `characterize` consumes three genuine levels only when
measuring self-step sensitivity; `compare-pair` can describe two saved participants
without fabricating a third reference resolution. Stochastic/unknown model
correspondence leaves qualified parity untested while raw differences remain
useful. Full event/population or stochastic qualification needs the corresponding
evidence, not a pre-event evaluator PASS.

The current evaluator requires paired IDs, one supported release slice,
identical observation clocks and position/velocity/charge columns. Its RMS
uses equal retained rows. It is not a weighted/time-integrated, position-only,
unpaired-ensemble or general staggered-release evaluator. Do not invent missing
v/Z, create artificial cohorts or force all objectives through that CLI. Use
the selected quantities and a small external aggregation against public results
or bound normalized exports; preserve the comparison record and metric meaning.

For this evaluator's qualified cross comparison, register independent
`self_sensitivity_budgets` for candidate and reference as well as cross
`physical_budgets`. Missing/exceeded self allowances or unproven actual
refined macro partitions leave qualified parity `NOT_TESTED`; a small finest
cross difference alone is insufficient. Its self-resolution gate consumes
completed public result manifests. CSV-only native execution/partition
authenticity remains unqualified by this tool. Read the actual report limits,
or use a source-bound native evaluation suited to the requested claim; do not
label a native CSV as a public result to bypass the gate. Empirical self
sensitivity still supplies neither an absolute error bound nor physical truth.

Evaluate the primary observable, initial meaning, relevant fields/RHS,
exercised events and numerical/statistical evidence at the depth required by
the claim. Diagnose the earliest unexplained layer. Normally test at most three
named discriminating hypotheses before reassessing; each retry adds evidence.
Preserve the first-divergence packet and eliminated causes. Attribute to
configuration, extraction/adapter, field production/transfer, physics/field
sampling, integration/events/boundaries, evaluator, reference uncertainty or
model-form limits only after the probe supports it. End-to-end error alone is
no core defect; an unresolved scoped conclusion remains useful.

For an improvement proposal, identify the failed use criterion and supported
owner, predict how the change affects the relevant observable, and specify an
independent verification and affected-case comparison. Fix extraction/settings
or representation at their owner; change core equations/events only with
independent evidence. No code change is needed solely to improve visual overlap
when purpose-level criteria already hold and no invariant is violated.

Deliver one comparison manifest/summary linking existing raw tables, cases,
results, commands/hashes and essential numerical evidence; add only missing
artifacts. For each requested claim state `PASS`, `FAIL`, `NOT_TESTED` or
`NOT_APPLICABLE`, scope and evidence. Raw quantitative differences remain
reportable with qualified parity `NOT_TESTED`. Use
[visualize-particle-trajectories](../visualize-particle-trajectories/SKILL.md)
when figures clarify fixed metrics and first divergence. Give one exact next
probe/export or owning module for actionable issues. COMSOL agreement is scoped
compatibility; physical validation requires independent physical evidence.
