---
name: inspect-comsol-model
description: Discover unfamiliar COMSOL MPH models, extract requested native settings and data, or verify reuse of existing extraction evidence before preparing or comparing a chamber-particles case. Use for field-only and particle-tracing models; this skill identifies the source problem and does not select solver approximations or claim trajectory agreement.
---

# Inspect a COMSOL model

Turn native observations into a reusable `extraction_record.md` with linked raw
readback and export receipts. Labels, remembered defaults, an MPH file hash,
and successful API calls alone do not establish which equations produced saved
data. Keep native observations, physical interpretations, and unresolved facts
separate. Existing chamber sizes, tags, IDs, gases, and time steps are examples.

## Select the work needed now

| Mode | Minimum inputs | Minimum result and exit criterion |
|---|---|---|
| `discovery-only` | MPH or available report/exports, requested region/question | Record source/access identity, components and solution links, active equation/field/particle inventory, and remaining uncertainty. Hand this to preparation once it can choose a physical scope; do not export every quantity first. |
| `required-exports` | Discovery record plus preparation's target export specification | Export only the selected geometry, primitives, releases, states or events; bind their dataset/member, units, basis, side, IDs and dependencies. Stop when those exports pass readback checks or give one exact missing-data request. |
| `verified-reuse` | Existing record/raw artifacts, present source and dependency paths, requested scope | Verify identities and hashes before reusing facts. Reuse unchanged evidence; invalidate affected facts after any source, dependency, scope, solution, reader or exporter change. |

If no mode is requested, start with `discovery-only` for an unfamiliar source,
or `verified-reuse` when an extraction exists. A request for a complete run
continues through [prepare](../prepare-comsol-field-case/SKILL.md), returns here
with the necessary export specification, then uses
[validate](../validate-comsol-trajectories/SKILL.md) for a requested comparison.
An inspection-only request ends with inspection. Do not add a mandatory report
bundle or simulation to that request.

For a focused saved-source review, inspect the selected release, particle
properties, frame and dependency bindings and actual initial rows. Reuse known
environment/access evidence; do not repeat license discovery, all-force extraction,
case preparation or a report bundle unless needed to answer that scope. Add the
small result to an existing record or return it directly when that is sufficient.

## Bind the environment and preserve the evidence

Locate the repository by `AGENTS.md` and the independent solver project. Read
the repository guidance and `particle_platform_redesign/AGENTS.md` in full.
Record the source revision/relevant local changes and actual imported solver
location/version when candidate preparation or results are in scope. A native
inspection alone needs no solver import. Never activate `old_code/`.

Reuse verified access evidence; discover access only for a new native observation.
Discover the exact COMSOL producer/reader versions, executable/API, licenses,
OS, and stored-solution access. Use one working native batch/Java, licensed
LiveLink, or authoritative report/export route. Check matching local help or
official documentation for uncertain properties; do not copy API names or
defaults across versions. Without native access, retain partial observations
and request the smallest missing export once. A report or binary/ZIP listing
does not decode a saved solution or replace the MPH identity.

Read [api-extraction.md](references/api-extraction.md) when using native access.
Load a copy/no-save model. Inspection must not solve, regenerate a sequence,
rebuild geometry/mesh, save over the source, `updateSolution()`, or refresh,
import or discard function data. Those operations can change the in-memory
payload even when the MPH file hash stays unchanged. A requested reproduction
uses a separately identified companion with explicit changes and readback.

Read [extraction-record.md](references/extraction-record.md) for dependency
capture and reuse. Distinguish embedded data, currently loaded data, present
external files, and data used by the historical solve. Record dependency
expressions/paths, resolution basis and resolved paths, content hashes, and
the evidence for the actually read state. MPH hashes alone cannot bind tables,
DLL/SO functions, scripts, CAD or other external inputs. The optional
[snapshot_inputs.py](scripts/snapshot_inputs.py) creates a file hash receipt;
it does not discover dependencies or prove which bytes COMSOL loaded.

## Observe before mapping

Store native typed values, original expressions/units, qualified feature paths,
resolved entity IDs and actual links in `raw/model_inventory.json`. Preserve
unavailable/nonfinite observations with validity and reason, while retaining
original exports; never substitute zero or an empty selection after an error.

Discovery enumerates components/DOFs, geometry/mesh, global and local definitions,
functions/materials, physics and nested features, studies, solvers, solutions
and datasets. Resolve the intended component, region, time/sweep member and
chain of field ownership before targeted readback. Distinguish enabled nodes,
study activation, solver inclusion, and evidence of use in the saved run.
Current settings may have been edited after solving; retain unknown provenance.

For particle models, read only relevant sections of
[particle-semantics.md](references/particle-semantics.md). Trace custom forces
and auxiliary equations through dependencies and assembled contributions.
Record all active mechanisms, even those preparation later excludes. A force
label or a gas-flow boundary name does not establish a particle law.

## Targeted export and handoff

Preparation selects supported physics and geometry/field representation, then
returns a specification containing selected dataset/solution/member, required
primitive expressions and units, coordinate basis, support/domain sides,
geometry connectivity/boundary identities, and any particle states needed.
Use that scope to avoid unnecessary exports or implicit model reduction.

For each export retain native and evaluated units, basis, side, time/parameter,
precision, owner IDs, exporter command/revision and hash. Check counts, unique
IDs, support, finite values, coverage and native evaluation at representative
points. Preserve raw values before SI conversion. Never join topology or
particles by rounded coordinates or row order, invent events, extend missing
tails, or flatten temporal/RF fields. Field-only sources need no particle export.

For requested result evaluation, also bind the population/time selection and
meaning of each exported statistic or plot: filters, release/secondary cohorts,
computational count versus physical weight, instantaneous versus time-accumulated
quantity, normalization, and available event/status storage. A trajectory plot
may omit particles or connect only saved positions. Export actual velocities
and charge when those quantities matter; a picture or position difference
cannot supply them. Validation selects the estimands and judges agreement.

Update the same extraction record and raw artifacts; link large data instead
of duplicating it. Preparation owns the model correspondence and numerical
plan. When preparing `semantic_inventory.json`, use only the existing seven
layers and preflight v2 in
`particle_platform_redesign/solver/tools/vv/comsol/README.md`. Actual MPH identity,
scoped settings/readback hashes and JSON pointers are distinct bindings.
`expected` and `observed` must refer to real artifacts; copying expectations is
not readback. Candidate bindings remain unresolved until candidate data exist.
Preflight classifies meaning, not transfer accuracy or successful execution.

Deliver the record, relevant raw readback/exports and receipts, repeatable
commands, and one next action per missing fact. Unresolved identity, equation,
unit, side, field ownership or historical provenance limits the dependent
claim; continue independent work and state the achievable scope explicitly.
