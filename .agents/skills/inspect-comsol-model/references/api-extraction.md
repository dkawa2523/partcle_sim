# Native API extraction without case-specific guesses

Use this when creating or adapting a narrow external exporter. The examples
below are discovery patterns, not a compiled universal MPH reader. Verify APIs
against the installed version and compile a small readback before a large
export. Keep COMSOL classes outside `src/chamber_particles/`.

## Access and two-pass discovery

Use `ModelUtil.loadCopy(tag, path)` where supported, or load a separate copy
with the local batch no-save option. Inspect settings/stored solutions without
calling a study or solver `run`, regenerating the solver sequence, rebuilding
geometry/mesh, or saving over the source. Inspect a companion separately when
the task requires an authorized new solution. Record executable/version,
command, source hash and exit status. A settings-only success is not a solved
data export. See the versioned [ModelUtil API](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/api/com/comsol/model/util/ModelUtil.html).

Executable version and Java compilation do not establish that native model
access or a required module license works. Check startup/run logs and the actual
new readback/export artifacts as well as process exit status: a COMSOL launcher
can return zero while reporting a license failure. An absent success marker or
required artifact leaves execution untested. On unavailable/expired licensing,
retain one failed-attempt receipt and use available saved evidence; repeating
the same invocation or fabricating a readback cannot improve that evidence.

Preserve the loaded payload as well as the source file. In inspection, do not
call `updateSolution()` / Update Solution: it can apply current definitions or
interpolate saved data after mesh/geometry edits without solving the new
equations. File hash equality cannot detect that in-memory change.
[Updating a Solution](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_solver.36.108.html)
Also do not call function `refresh()`, `importData()`, `discardData()` or `run()`;
these can reload, replace, discard or train input data. Export/read the existing
payload where supported and retain unavailable readback explicitly.
[Function API](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_general.47.34.html)
Creating temporary result evaluation/export nodes on the inspection copy is
acceptable when it leaves the saved solution and input payload unchanged.
Evaluation of an external function executes its user library; first inspect
the library's identity and execution requirements, and isolate evaluation when
side effects are possible. Do not treat a successful call as historical binding.

The prohibition applies to reading an existing reference. Building an
authorized new companion may require `importData()` or other input operations.
Record those changes, snapshot the inputs, inspect the loaded readback and
new solution afterward, and keep the companion separate from the original.
Do not “repair” the saved reference in order to make extraction succeed.

Enumerate native tags before selecting them. Representative Java patterns used
by the repository's external inspection examples are:

```java
model.component().tags();
model.component(ctag).physics().tags();
physics.prop().tags();
physics.feature().tags();
physics.feature(ftag).feature().tags(); // recursively inspect children
model.study().tags();
model.study(stag).feature().tags();
model.sol().tags();
model.sol(soltag).feature().tags(); // recursively inspect solver children
model.result().dataset().tags();
```

Preserve the native API hierarchy as observed: a dataset tag normally belongs
to `model.result().dataset(tag)`, while its expressions may contain
`root.comp1.*` variable scopes. Do not invent a component-qualified dataset
path from an expression prefix. Record the actual dataset/solution links,
parameter names and one-based member indices; an unknown outer tuple remains
unknown rather than being reconstructed from a suggestive tag or filename.

Also enumerate the selected component's geometry, meshes, variable groups,
functions, materials, named selections, and multiphysics couplings, including
global definitions. Preserve component-qualified paths. Two interfaces can
define identically named variables with different scopes.

Trace file-backed definitions and geometry inputs to the dependency ledger in
[extraction-record.md](extraction-record.md). Inspect source mode, interpolation
and extrapolation, argument/value units, library initialization, and nested
function dependencies. A filename/property readback does not identify the
loaded table or the library bytes; report the distinction rather than invoking
a refresh to hide it.

Read `getType()`, `label()`, activation, property groups and nested features.
Use `properties()`, `hasProperty(name)` and `getValueType(name)` before choosing
the corresponding boolean/integer/double/string/array/matrix getter. Complex
or version-specific map types need their documented accessor. Record an
unsupported getter/type as unavailable, with the error and raw type. Do not
convert every property to string or catch an error and write `[]`, `0`, or
`false`. `getAllowedPropertyValues()` can return null for an unrestricted
property. [PropFeature API](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/api/com/comsol/model/PropFeature.html)

## Effective selections and equations

Read both named selection references and resolved entity IDs, entity dimension,
geometry/component scope and input/output selection distinction. Use
`selection().entities(dim)` with the actual dimension: a 2D boundary has
dimension 1; a 3D boundary has dimension 2. Empty selection and failed selection
readback are different observations. Capture material and force domain sides,
internal interfaces, overlapping boundary features and conditional behavior.
[Selection API](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/api/com/comsol/model/Selection.html)

Check feature `isActive()` and the chosen study's activation/advanced disable
settings for physics, nested features, variables and couplings. Record actual
solver dependent-variable and background-solution choices. Property names and
types change by version; discover them rather than assuming a fixed
`activate` string array. Distinguish requested study settings from customized
solver settings. [Studies and Study Steps](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_solver.51.59.html)

For custom force/charge terms, recursively retain expression dependencies,
parameter/function/material sources and Equation View rows when available.
`featureInfo(...).getInfoTable("Expression", "recursive", "all")` provides
expression information; `Shape`, `Weak`, and `Constraint` information can also
be relevant. Preserve raw rows/headers and version instead of guessing force
variable names or fixed column offsets. This is equation contribution
information; it does not prove the stored solver used that contribution.
[FeatureInfo API](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/api/com/comsol/model/physics/FeatureInfo.html)

## Solution, dataset and evaluation ownership

For each candidate dataset record type, physics/geometry/solution references,
selection, transformations, time/parameter selections, and whether it is a
derived dataset. Walk through a cut, filter, revolve, join or particle dataset
to the underlying solution. A plot's selected dataset need not be the desired
field source or particle cohort.

For a solution inspect type, available solution count, parameter names/values,
associated mesh and `getSolutioninfo()`. `getPVals()` is not generically a time
array. Resolve time/sweep/inner/outer members through their names and units.
[Solution information](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_solver.51.08.html)

Use versioned `SolutionInfo` methods to resolve outer/inner solution tuples and
mesh changes. Document indexing: many solution numbers are one based while
`getISol()` returns zero based multisolution/local indices. An info object's
`isValid()` is not historical proof that current model settings produced the
saved solution. Record provenance as unknown if no reliable run record binds
them. [SolutionInfo API](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/api/com/comsol/model/SolutionInfo.html)

Keep original expressions and evaluated values/units separately. Parameter
evaluation (`model.param().evaluate`/`evaluateUnit` in local examples) applies
to parameter expressions. Evaluate position-, time-, field-, or particle-
dependent expressions with the specified dataset, solution tuple, point/entity
and time. Check dimensional units after evaluation. Do not use a parameter
evaluator to guess a spatial expression or interpret a complex RF phasor as an
instantaneous real field.

## Authoritative exports and their checks

Export geometry/mesh topology with element order/connectivity, native node and
entity IDs and boundary ownership. A rendering mesh or filtered plot is not
the solution mesh. Retain native geometry before any tessellation; downstream
preparation owns the quantified curved-wall approximation and mesh changes.

Export field data at nodes/cells or specified evaluation points with explicit
IDs, side and basis. Retain time dependence, parameter identity, derivative
recovery method and missing/support mask. Evaluate a small set of interior,
near-boundary, interface and axis points directly in COMSOL to check exporter
units, orientation and selection. A rounded-coordinate nearest-neighbor join
can create false values even when a plot looks plausible.

For particle data verify the actual particle dataset's position DOFs and
identity/statistics expressions before querying them. Export release identity,
time, states and available event/status evidence. Distinguish a native missing
quantity from a parser error. Treat unavailable rebound event states as
unobserved; changing storage controls only affects a new companion solution.

Before handing off, check counts, ID uniqueness/coverage, entity dimension,
finite values on required support, extents and basis, selected sweep/time
coverage, and raw-to-normalized hash lineage. Test one export readback before
launching a costly full run.

## Reuse boundaries

The local Java files `InspectM3VModels.java`,
`InspectM3C2ModelSemantics.java`, and `ParticleRunReadback.java` under
`particle_platform_redesign/solver/tools/vv/comsol/comsol/` illustrate native
readback. They use specific tags, expressions, 2D selections and source names.
In particular `semanticGroup(tag)`, force-tag filters, and hard-coded particle
properties are not generic knowledge. Reuse a verified access pattern, not
their case assumptions. A new exporter should accept its selected model/scope
explicitly and emit evidence, leaving model selection and numerical verdicts
to preparation and evaluation.
