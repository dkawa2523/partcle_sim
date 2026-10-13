# Extraction and handoff record

Use this form for `extraction_record.md`. Keep values and raw evidence in
linked artifacts rather than duplicating large tables. A missing value is
“unavailable: reason / next observation”, never an assumed default. This form
does not replace the canonical case or preflight v2 schema.

## Scope and environment

- Mode (`discovery-only`, `required-exports` or `verified-reuse`), requested
  minimum deliverable, and the criteria for handing off or ending this mode.
- Requested question/region, source MPH path and SHA-256 before/after.
- Repository revision plus relevant local changes; actual imported solver
  location/version and uv lock identity.
- COMSOL producer and reader versions, OS, modules/access route, exporter
  revision/command; stored data available or new companion run required.
- Chosen component/physics, geometry, study/step, solver/solution, dataset,
  sweep values, time range, and explicit chain of links between them.
- Evidence tying saved data to solved settings; unknown/stale binding if absent.

## Inputs outside the MPH and the actually loaded state

Discovery identifies file-backed functions/tables, shared libraries (DLL/SO),
MATLAB or other external functions, material databases, imported geometry/CAD,
and any nested data they use in the selected expression dependency graph. Read
only the dependencies relevant to the requested scope; an unrelated inactive
feature is not a blocker. Keep this ledger in the existing record and raw
inventory, rather than creating a second model definition.

| Definition/path | Source mode and resolution | Present bytes | Loaded/embedded payload evidence | Saved-solve binding and affected quantities |
|---|---|---|---|---|
| Qualified function/geometry tag and original expression/path | Embedded/model resource, table, result table, external file/library; model directory/resource/search path and resolved absolute path | SHA-256 + size, or missing/unreadable reason | Raw property/table/payload export and hash; present/unknown loaded state; no refresh performed | Historical input receipt or unavailable; list geometry, primitive, postprocessed or equation inputs affected |

For relocatable handoff, give the path relative to the handoff root as well as
the original native and resolved environment paths. Record library platform/
version, initialization string and known indirect resources. A DLL/SO hash
does not cover an undocumented runtime lookup or remote service. Mark that
dependency unresolved rather than claiming complete input identity.

Separate these observations explicitly:

- The bytes currently on disk, and the file COMSOL resolves in this environment.
- The embedded or cached bytes currently loaded in the model, if observable.
- The bytes used by the saved solve, if bound by a reliable historical receipt.

For an embedded interpolation table, hash an exported/readback representation
of the loaded data and its settings. For external/cached data, the source
filename and current disk hash alone cannot prove loaded data equality. When
the version has no suitable loaded-state readback, record unknown and use only
unaffected stored primitives or a separately identified reproduction. Keep the
original saved model unchanged. `refresh()` / `importData()` is a new input
operation, not an observation of the old input.

### File snapshot helper

Use [snapshot_inputs.py](../scripts/snapshot_inputs.py) when hashing the same
explicit source/dependency/exporter files repeatedly. It uses Python's standard
library and reads literal relative paths under an explicit root (including
explicit `../` dependencies). It does not search directories, execute COMSOL,
read library dependencies automatically or certify a model. Invoke it using
the independent solver uv environment, or an available Python when the solver
is not installed:

```console
python .agents/skills/inspect-comsol-model/scripts/snapshot_inputs.py --root HANDOFF_ROOT --output NEW_RECEIPT model/chamber.mph dependencies/gas.txt exporters/readback.java
python .agents/skills/inspect-comsol-model/scripts/snapshot_inputs.py --root HANDOFF_ROOT --output NEW_RECEIPT --compare OLD_RECEIPT model/chamber.mph dependencies/gas.txt exporters/readback.java
```

Every output must be new and distinct from all inputs and the previous receipt.
Comparison reads the old receipt without changing it; paths/hashes are compared
relative to the explicit root so the bundle may be relocated. Exit 0 means all
listed files were read stably and either this is an initial receipt or the
previous path/content set agrees; exit 1 means the set/content changed; exit 2
means an input is missing, unreadable, non-file, or changed during reading.
Separate current disk identity from COMSOL loaded/historical identity even
after an exit 0. Retain the receipt hash in the extraction record.

## Required facts

Use a row per distinct physical fact. Suggested columns:

| Fact | Native observation + artifact pointer | Interpretation + source | SI conversion / candidate meaning | Unresolved / next observation |
|---|---|---|---|---|
| coordinate/DOFs | | | | |
| selected domain/side | | | | |
| background field owner | | | | |
| particle formulation | | | | |
| enabled term and expression | | | | |
| particle/release authority | | | | |
| boundary group/law | | | | |
| numerical/output controls | | | | |

Expand only the relevant rows with the following information:

**Geometry:** XY/RZ/3D, length unit, origin and transform, out-of-plane DOFs,
axis treatment, stationary/moving geometry; domain/boundary/entity IDs and
selections, material assignment; element type/order/connectivity/orientation,
curved wall representation, ownership of internal interfaces and periodic
pairs. Record actual mesh version associated with the selected solution.

**Field inventory:** primitive expression, unit and sign, vector component
order/basis, gas/species identity and closure, static/time/phase convention,
mesh/layout/node or cell IDs, support domains and side, evaluation precision.
State if a derivative is native, recovered, differentiated after resampling,
or analytically derived. Preserve the producer field and conversion lineage.

**Equation ledger:** physics type, dependent variables/formulation, active
feature path, study/solver activation, selection and overlap/override,
expression or built-in law, constants and their sources, material/field
dependencies, force versus acceleration, auxiliary charge equation and units.
Include coupled feedback, collisions, secondary emission, variable size/mass,
relativistic/out-of-plane motion, or other mechanisms present in the model
even if they will be excluded from the comparison scope.

**Particle/release ledger:** stable native identity and cohort, actual release
time/location/velocity, mass specification and evaluated mass, drag diameter,
electrical radius, displaced volume, charge specification and value,
weight/rate meaning, distributions and seed semantics, source selection and
surface launch convention. If mass is derived from density and size, record
that derivation once; candidate inertia remains `mass_kg`.

**Boundary ledger:** actual entity IDs, effective active feature and conditional
priority, native wall/outlet action, pre/post-state rule, reflection probability
and angular/speed law, restitution or temperature, periodic mapping,
secondary particles, moving wall. Gas inlet/outlet, material name, and electric
potential do not determine the particle law.

**Numerical ledger:** formulation/DOFs, actual integrator and step control,
initial/max/fixed internal step, relative/absolute tolerances and state scaling,
output times and storage interpolation, wall interaction accuracy/limit,
released/status/event data storage, Brownian amplitude/step convention and RNG.
Separate requested study settings from actual solver settings and saved times.

**Result observations (when evaluation is requested):** selected expressions,
plot/dataset filters and time window; release/secondary cohorts; computational
count versus physical weights/rate; count/density/flux normalization and
area/volume measure; instantaneous versus time-accumulated statistics; actual
event/status storage, first/last observations and missing-row coverage.
Bind these to the native settings and raw exports. A saved trajectory image or
an accumulator label alone does not identify the underlying estimand.

## Export receipt

| Artifact | Source dataset/solution/sweep/time | Expressions/units/basis/side | Identity/join and precision | Command/revision | SHA-256 |
|---|---|---|---|---|---|

Include geometry/primitive/release/trajectory/event exports only when required.
Check counts, duplicate IDs, unit dimensions, extent, native-to-export identity
coverage, absent domains, nonfinite values and times. Mark physical disappearance
separately from exporter failure and unobserved states. Record first/last valid
observation per identity; never extend a disappeared tail.

## Downstream readiness and continuation

Summarize the existing preflight layers: `coordinate_dof`, `formulation`,
`field_representation_owner_recovery`, `source`, `boundaries`, `models`,
`integration`. Reference raw evidence and scope/mapping; let the existing
preflight produce its classifications when bindings exist.

Record what can be prepared now, what comparison is possible, what cannot yet
be claimed, and the smallest missing observation per blocker. Include a single
next action, completed commands/artifact hashes, and superseded artifacts.
Another agent should resume from this record rather than infer context from
the conversation or rerun the entire extraction.

In `discovery-only`, preparation receives the inventory and returns a targeted
export specification after choosing physical scope/representation. Keep that
specification in this record: dataset/solution/member, expressions and units,
coordinates/basis, required support and domain sides, geometry/boundary IDs,
identity/state/event coverage, and purpose of each export. Inspection does not
guess a solver model merely to choose quantities to export.

In `verified-reuse`, compare the current MPH, explicit dependencies, selected
solution/member and scope, reader/exporter identity and existing artifact
hashes before accepting facts. Any changed dependency triggers review of its
affected facts even when the MPH hash and native string paths are identical.
An unchanged, verified embedded payload or directly stored primitive can remain
usable when an external file changed, but only with readback showing that the
quantity is independent of the changed file. A missing external dependency
blocks dependent postprocessing or a new solve; it need not invalidate an
already locked, independent saved primitive. Record this decision explicitly.
Re-read or re-export the affected facts, not the entire model indiscriminately.
Changed COMSOL versions require verification of semantics relevant to the
selected scope; do not silently update or re-solve an older reference.
