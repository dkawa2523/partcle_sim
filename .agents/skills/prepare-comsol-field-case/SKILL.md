---
name: prepare-comsol-field-case
description: Map COMSOL geometry, fields, and particle requirements into chamber-particles inputs, select supported or justified approximate physics, and choose independent numerical settings. Use for a preparation plan, checked inputs, smoke run, or requested scientific execution. Trajectory agreement is evaluated by validate-comsol-trajectories.
---

# Prepare a COMSOL chamber case

Turn source evidence and the requested physical problem into the smallest useful
preparation deliverable. Case tags, boundary IDs, dimensions, seeds, and step
sizes from another chamber are examples, never new-case defaults.

## Select the requested scope

Infer the scope from the user's request; do not ask merely to select a mode.
If only preparation is requested, finish at checked inputs. A field-only source
does not specify particle releases, properties, charging, or particle walls.
Use supplied requirements or declared physical assumptions, and bundle only
the unresolved choices that materially change the problem.

| Scope | Required input | Minimum useful output and stopping point |
|---|---|---|
| Plan | Source evidence or its access limits, intended physical question | Model mapping, targeted export specification, numerical intent, and exact missing evidence; no execution claim |
| Preparation only | Required geometry/fields and particle/boundary requirements for a runnable case | Canonical HDF5/YAML, hashes, conversion/geometry/field checks, shared execution-preparation `check` result, numerical plan; particle execution and accuracy remain untested |
| Smoke | Checked inputs and a bounded representative run | Separate smoke derivative, command/result identity, resolved models and failure/lifecycle counts; no accuracy or agreement claim |
| Scientific execution | Checked case, relevant observables/error budget and run scope | Requested run and registered numerical experiments, measured resolution evidence and limitations; route external comparison to validate |

When only background fields are requested, deliver checked field/geometry
artifacts and their provenance; retain a draft particle-case plan if necessary.
Do not invent a particle population to present those fields as a checked
trajectory case. Genuine resolution studies support numerical qualification;
they are not prerequisites for planning, a wiring smoke or saved-result visual/
practical comparison. Use the numerical plan's appropriate independent evidence.

## Bind evidence to the current solver

For source work locate `particle_platform_redesign/solver/pyproject.toml`, read
the applicable `AGENTS.md`, and use that independent uv project with
`uv run --locked`. Normal input clients and CLI also ship in the installed
wheel; source checkout is needed only for development/native V&V work.
Reuse verified source/export records; discover COMSOL
access only for missing native observations or authorized new reference runs.
Record package/source revision and dirty state, lock identity, case schema, and
the selected model/numerical revisions. Read the chosen model sections of
`solver/docs/physics_models.md`, schema in `solver/docs/case_format_v3.md`, and
relevant `solver/docs/numerics.md`. Resolve contradictory historical text against
current selectors, preparation gates, and run readbacks before claiming support.

Reuse a fresh extraction record from
[inspect-comsol-model](../inspect-comsol-model/SKILL.md). Invoke it when source
equations, selections, solution provenance, units, or identities are unfamiliar
or missing; a complete matching record needs no repeat inspection. Preparation
is an iterative **discovery → model mapping/export specification → targeted
extraction** process. Discovery may be enough to select a model without yet
having its field exports. Request the required primitives after that decision,
then recheck the mapping; do not demand every possible COMSOL field up front.

## Map equations and request only necessary exports

Use [model-mapping.md](references/model-mapping.md) for the model traps and
geometry/field gates. Maintain one small mapping record with source feature and
evidence, actual equation/parameters, solver category/model/revision or omission,
`EQUIVALENT`/`APPROXIMATE`/`UNSUPPORTED`/`UNRESOLVED`, required primitives,
changed assumptions, regime evidence, and affected observables.

Model names are insufficient to establish equivalence. Match equations,
conductivity and mean-free-path conventions, particle authorities, species,
state coupling, degrees of freedom, release, and wall meaning. A supported
slice, common-model companion, or useful approximation has its own scope and
identities; preserve the original inventory. An approximation needs a physical
reason and planned sensitivity or error allowance, not tuned coefficients.

For particle releases read [initial-conditions.md](references/initial-conditions.md)
when vector, energy, angular/thermal distribution, surface measure, correlation or
frame conversion needs interpretation. Check actual realized velocity as well as
position. Source generation belongs outside the core; current canonical tables
already retain per-particle vectors, release times and property authorities.

Write a targeted export specification for each selected primitive and topology:
source expression and solution/dataset/parameter/time selection; domain and
interface side; SI unit/components/basis; topology IDs/connectivity and sampling
locations; spatial/time accuracy checks. Include actual particle IDs/releases
and source/wall parameters only when particle preparation needs them. Missing
derivatives require an explicit recovery method; an unrelated diagnostic or
similarly named COMSOL variable is not a substitute.

Classify field production as direct export, independently checked reconstruction,
explicit reduced-field model, or unsupported. The installed
`chamber_particles_tools.csv_import` format-4 client converts explicitly
selected static nodal XY/RZ fields on one volume domain;
it records units, basis, producer meaning and explicit axis transforms. Read its
README/config schema before use. `provider: comsol` requires native dataset,
solution and member lineage; `provider: csv` handles caller-owned numeric CSV
without inventing native provenance. It is not an MPH reader,
time/cell/discontinuous field converter or physical-model selector. Other
representations need an explicit producer using the canonical writer, without
core branches. Read the reduced electrostatic builder's README before choosing
its gas-only RZ closure; full-material/floating-conductor V/Q capability is not
implemented by this builder. Neither path chooses
the YAML physics models or proves COMSOL-field equivalence.

Use the scope-aware preflight interpretation in
[model-mapping.md §6](references/model-mapping.md#6-interpret-meaning-preflight-in-the-requested-scope).
Missing native evidence leaves a draft and a specific export request. Field-only
preflight exit code 2 can mean that no comparison question was requested;
inspect its JSON and relevant layer blockers before deciding whether to proceed.

## Convert and check

Preserve source files/dependency identities, stable IDs, accessible domains,
holes/openings, boundary/source groups, units and coordinate basis. Check topology,
field support and time coverage, nodal correspondence, independent off-node
values, source containment, and conversion errors as specified in the mapping
reference. Distinguish a material particle wall from a transparent domain
interface. A shared-node P1/Q1 field cannot retain a jump that particles cross;
selecting one side is valid only when that is the accessible domain. Otherwise
register unsupported representation or a quantified approximation.

Use `chamber-particles models` for current model requirements and `init` for
editable installed examples, not a chamber-specific template. Publish through
`chamber_particles_tools.inputs.create_case(bundle, settings, output)`
or CLI `create RECIPE -o OUTPUT`. The bundle owns available source tables;
settings select actual releases. Recipe format 2 binds an `input_bundle`,
the complete `sources` table set and normal case `settings`. Each table is an
explicit CSV/configuration or an `existing_table` reference; unlisted tables
are removed. The publisher owns data
path/hash and calls public `load_case` and `check_case` before publication.
Normal CSV/cache/electrostatic producers publish `bundle.h5` with `report.json`
in one fresh output directory; use that bundle path in the recipe. Runnable
examples publish their editable sources/recipe in the same directory commit.
It can add the first source to a bundle with no existing particle tables.
Source format 2 supports internal and surface realized CSV rows; independent
properties are explicit unless `inherit_from` requests an exact-ID join.
Canonical tables own physical facts, YAML owns model/wall/numerical choices,
and worker count is an API/CLI resource. Never infer inherited physical
authority from equal integer IDs alone.
If an external producer declares generated coefficient groups, the normal
publisher checks that record against the current primitives, layout/clock,
parameters, declared particle properties and all output fields. Keep that
record through input transformations; do not remove it to accept stale fields.
Independent prescribed fields remain valid without invented dependencies.

Run `chamber-particles check <case.yaml> --workers N` for a preparation-only
deliverable. It uses the same model/support/topology/stability/memory prepare
as `run`, without workers, writer or particle updates. This is readiness for
the requested worker count, not runtime completion or accuracy evidence.

## Choose numerics and execute the requested scope

Read [numerical-plan.md](references/numerical-plan.md) for integrator choice,
timescale estimates, deterministic convergence and Brownian uncertainty.
Choose `dt_s` from mapped equations and field/particle variation. Keep it separate
from COMSOL output/internal steps, field snapshots, releases and observations.
Acceptance/stability gates do not certify accuracy. Record a pilot step and
unmeasured budgets when execution was not requested.

For smoke, preserve the requested case and create a separate derivative with
population/time/output burden reduced and changes recorded. HDF5-realized
populations need a separate canonical subset with stable IDs; YAML cannot subset
a table. Preserve models, forces, charge, contact and wall laws. Use fresh paths:

```console
uv run --locked chamber-particles check <case.smoke.yaml>
uv run --locked chamber-particles run <case.smoke.yaml> -o <smoke-result> --workers 1
uv run --locked chamber-particles inspect <smoke-result>
```

`python -m chamber_particles` is the same installed dispatcher. Normal result
schema 5 includes immutable geometry/semantic context; `analyze`, `export` and
`view` reuse RESULT alone through `ResultView.read_context()`. Input case/data
remain schema 3. No old-result reader or silent migration is provided.

Inspect actual resolved model/numerical revisions, lifecycle and
`failure_reason_counts` even after a successful CLI exit. Diagnose input,
conversion, field production, core, or evaluator ownership before modifying
anything; do not suppress a failure with a clamp, nudge, or relaxed threshold.
For scientific execution, run the authorized case and registered resolution
matrix, retain commands/input/result identities and measured budgets, and stop
with a scoped conclusion when a demonstrated limit remains.

Hand off only the evidence produced in the requested scope: mapping/export
specification, canonical artifacts/hashes, conversion and geometry/field checks,
numerical plan, commands/results, unresolved assumptions and missing evidence.
Use [validate-comsol-trajectories](../validate-comsol-trajectories/SKILL.md) for
external trajectory evaluation. A field preparation or smoke is a useful
deliverable; its success does not establish accuracy or COMSOL agreement.
