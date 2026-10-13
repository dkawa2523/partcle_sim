# Map source meaning to a solver case

This is a decision aid for the currently checked source, not a second physics
catalog or a template with chamber defaults. Read the selected model's complete
section in [physics_models.md](../../../../particle_platform_redesign/solver/docs/physics_models.md)
and its resolution in
[physics/catalog.py](../../../../particle_platform_redesign/solver/src/chamber_particles/physics/catalog.py).
Recheck when the solver revision changes. COMSOL feature definitions and their
versioned official references belong to
[inspect-comsol-model/particle-semantics.md](../../inspect-comsol-model/references/particle-semantics.md).

## 1. Model choice is an equation decision

For each source feature, compare the actual force/rate expression at the
selected study, not its label. Record whether it contributes force [N],
acceleration [m/s^2], number collection rate [1/s], electric current [A], or
charge rate [C/s]. A user-defined force may already include mass division, gas
drag, or gravity. Check double counting before enabling another category.

Choose one of these outputs:

| Status | Required evidence and resulting claim |
|---|---|
| `EQUIVALENT` | Same equation and evaluated coefficients, primitive meaning, units, species, state coupling, dimensions, release, and boundaries. Numerical method/resolution is assessed separately; unmeasured convergence does not change equation correspondence. |
| `APPROXIMATE` | Explicit changed closure/regime/degree of freedom, justification, affected observables, and a sensitivity or model-error allowance. The run tests a useful approximation; mismatch is not by itself a core defect. |
| `UNSUPPORTED` | Required state, force, geometry, or coupling is outside the current capability. Offer a supported slice, common-model companion, or native-versus-approximation comparison with a separately registered scope. |
| `UNRESOLVED` | A necessary equation, evaluated value, domain side, or unit is unavailable. Preserve the extract; name the exact missing evidence. Do not complete the mapping by guessing. |

Use one revision per category. Omit disabled categories; `charge` must always
be explicit. There is no automatic regime switch, hidden mixture closure,
optional force silently ignored, or producer-specific model in the core.

## 2. Primitive and regime index

Notation in the table is only a compact index: gas `u` [m/s], `rho` [kg/m^3],
`T` [K], `mu` [Pa*s], `lambda` [m], `k` [W/(m*K)]; species `n` [1/m^3],
temperature [K], thermal voltage [V], velocity [m/s], mass [kg]. Every vector
must use the canonical case basis; every scalar has `(value)` / `scalar`.
The exact field names are case-selected bindings, not automatic aliases.
Required positive primitives and all detailed applicability conditions are
owned by the catalog and physics source, including conditions on continuous
paths rather than only saved samples.

| Solver category / model / revision | Required canonical primitives | Selection trap or essential regime check |
|---|---|---|
| `charge / fixed` | No field; realized `charge_number` | `Z` is the charge number, `q=eZ` [C]. Fixed is not zero and not continuous plasma equilibrium. YAML is `{model: fixed}`; the resolved manifest records `fixed_charge_v1`. |
| `drag / epstein_linear / epstein_linear_v1` | gas u, rho, T, lambda; constant molecular mass and delta | Single gas Maxwellian, radius Kn `lambda/(d_drag/2)>=10`, slip/mean-thermal-speed <=0.1; delta in [1,13/9]. Low-density gas is insufficient evidence. |
| `drag / epstein_linear / epstein_linear_effective_gas_sensitivity_v1` | Same fields plus producer-certified effective Maxwellian and explicit `maximum_speed_ratio` | Same linear formula under an explicit pseudogas approximation; not a species sum. Declared speed ratio is <=1 and is a rejection gate, not a clamp. |
| `drag / epstein_finite_speed / epstein_finite_speed_maxwell_mixed_equal_temperature_v1` | gas u, rho, T, lambda; molecular mass, diffuse reflection fraction, maximum speed ratio | Free molecular, isolated nonrotating sphere; surface T equals gas T, Maxwell mixed reflection. Finite-speed is a different equation; do not enlarge linear Epstein applicability to imitate it. |
| `drag / stokes_cunningham / stokes_cunningham_allen_raabe_air_v1` | gas u, rho, mu, lambda | Air correlation and compatible mean-free-path convention; radius Kn in [0.03,7.2], particle Reynolds <=0.1. A COMSOL Stokes/slip label alone does not establish these coefficients or the Kn convention. |
| `electric / coulomb / electric_coulomb_v1` | electric vector E [V/m] | Force is `Z*e*E`; actual stage charge and `mass_kg` set acceleration. A magnetic/Lorentz term, electric potential alone, or an RF time average is not this complete input. |
| `gravity_buoyancy / standard / gravity_buoyancy_standard_v1` | gas rho; explicit gravity vector [m/s^2] | Uses separate displaced volume and mass. All RZ cases require radial gravity exactly zero; do not identify radial acceleration with uniform gravity. |
| `charge / plasma_continuous / oml_stationary_maxwellian_debye_huckel_v1` | electron and positive-ion n and T; positive-ion velocity; constant single positive-ion mass | Actual evolving Z; absorbing equipotential sphere, one-way collisionless unmagnetized single-ion Maxwellian; `a/lambda_D<=0.1`, relative ion drift/mean-thermal-speed <=0.1. Ion velocity is a drift gate, not an effective-temperature substitution. |
| `charge / plasma_continuous / oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1` | Same species fields plus declared maximum ion drift ratio | Shifted ion Maxwellian, nonpositive potential branch and initial Z<=0, certified nonpositive equilibrium/invariant; not arbitrary drift, positive-charge, multi-ion, or emission physics. |
| `charge / plasma_continuous / aggregate_relative_drift_regularized_two_current_v1` | electron/positive-ion n and thermal voltage, ion velocity, effective positive-ion mass, screening length [m] | Explicit aggregate regularized reference equation and declared maximum relative ion speed. Do not confuse it with species-resolved OML or recompute its supplied screening from another closure. |
| `charge / plasma_continuous / aggregate_relative_drift_regularized_three_current_v1` | Two-current primitives plus negative-ion n, thermal voltage, velocity, effective negative-ion mass | Negative-ion current is an additional term; source negative-ion diagnostics alone do not prove the source charging law contains it. Zero n is allowed for negative ions; other required fields still undergo checks. |
| `ion_drag / barnes_collisionless / barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1` | electron/positive-ion n and T, ion velocity, ion-neutral mean free path; constant ion mass | Collection plus orbital model, nonpositive particle charge, weak-coupling/collisionless gates and positive Coulomb log. A similarly named source may use another screening, collection speed, cutoff, or direction. |
| `ion_drag / screened_collection_orbital / relative_flow_screened_collection_orbital_aggregate_ion_v1` | positive-ion n, thermal voltage, velocity, effective mass, screening length, ion-neutral mean free path | Explicit aggregate sensitivity equation directed by relative ion flow; declared relative-speed bound. Not automatically the Barnes revision. |
| `ion_drag / image_orbital_sensitivity / electric_field_directed_image_orbital_sensitivity_v1` | positive-ion n, thermal voltage, velocity, effective mass; electron thermal voltage, screening length, E | Explicit electric-field-directed image/orbital sensitivity; not relative-flow drag. If Coulomb is enabled, use the same E authority. |
| `thermophoresis / waldmann_gallis / waldmann_gallis_free_molecular_single_species_heat_flux_v1` | gas u, T, lambda; translational conductive heat-flux vector [W/m^2], molecular mass | Single gas, free molecular and low drift. Total, radiative, convective, plasma-species, or solid heat flux is not this primitive. Fourier reconstruction needs translational conductivity and its gradient/regime evidence. |
| `thermophoresis / waldmann_gallis / waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1` | Same inventory plus certified effective gas/heat flux and explicit maximum speed ratio | Explicit pseudogas sensitivity; native and effective versions cannot be mixed across linear Epstein/Waldmann categories. |
| `thermophoresis / talbot / talbot_cross_regime_radius_knudsen_v1` | gas T, grad T [K/m], rho, mu, defined k, lambda; particle conductivity and three explicit slip/exchange coefficients | Radius Kn `lambda/(d_drag/2)`, not diameter Kn. Original unconverted coefficients use `lambda=2*mu/(rho*c_bar)` and translational k for polyatomic gas. Follow the convention conversion below when sharing lambda; native total-k variants are distinct. Broad equation range is not mixture/near-wall validation. |
| `dielectrophoresis / quasistatic_spherical / quasistatic_spherical_gradient_e2_v1` | grad of mean squared E [V^2/m^3]; medium permittivity, real CM factor, point-dipole radius ceiling | DC squared magnitude versus RF mean/RMS convention must be explicit. Particle electrostatic radius controls the dipole term. Complex-frequency CM, traveling-wave, or magnetic effects need another model. |
| `lift / rarefied_vorticity_sensitivity / rarefied_vorticity_sensitivity_rz_v1` | RZ gas u, rho, lambda, signed azimuthal vorticity [1/s]; explicit lift coefficient | RZ meridional/no swirl and free molecular; sensitivity law, not Saffman. A vorticity magnitude loses force direction. |
| `lift / saffman / saffman_unbounded_creeping_shear_v1` | gas u, rho, mu, lambda, signed out-of-plane vorticity [1/s] | Low Reynolds continuum, unbounded shear and hierarchy conditions; Stokes-Cunningham or no drag only. Not near-wall lift or Epstein-compatible. |
| `noise / inertial_langevin_fdt / inertial_langevin_fdt_epstein_linear_midpoint_2d_v2` | Shares selected native/effective linear Epstein primitives; explicit base tree depth, optional max depth | Inertial 2-D XY/RZ midpoint frozen-coefficient OU model. Requires `ou_langevin`; not overdamped, arbitrary nonlinear drag, isotropic 3-D, or the same Brownian path as COMSOL. |

The catalog additionally requires shared neutral-gas fields/constant molecular
mass to match wherever categories share a primitive. Charge and ion drag must
share the same ion background and compatible model family. Fixed charge can
be used where a compatible continuous-charge family is absent, but that is an
explicit model change. Noise requires supported linear Epstein, so it cannot
be combined with Saffman or Stokes drag by relabeling a model.

For every selected model, list each required field's exact unit/components/
basis/layout, source expression, conversion formula, domain side, and snapshot
selection in the mapping ledger. Check dimensionless regime margins over the
input field ranges and a conservative particle-state range. After execution
check path-related applicability failures; finite saved samples alone cannot
prove the whole path satisfies the model.

### Particle temperature and gas-surface energy exchange

A disabled native `Compute particle temperature` option means that the
auxiliary thermal equation is not solved; it does not establish a particle
surface temperature equal to the gas temperature. The current finite-speed
equal-temperature mean law uses prescribed `T_surface=T_gas` as an explicit
closure. Report that assumption when the native thermal state is absent.
COMSOL's temperature equation requires particle heat capacity and heat sources
([COMSOL 6.4 temperature theory](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.52.html)).

High relative kinetic energy can partition into translation and internal heat.
Do not infer heat capacity, composition, temperature rise, or phase stability
from density/diameter alone, or equate gas diffuse fraction with energy
accommodation. For a physical thermal claim obtain initial/internal temperature,
heat capacity and the surface energy/heat-transfer closure; otherwise keep
temperature-dependent drag/noise/thermophoresis explicitly conditional. A mean
force or collision-moment check is not thermal closure validation, and smaller
integration steps cannot supply these missing inputs.

### Conductivity and mean-free-path conventions

Read the Talbot input definition in the physics owner rather than inferring
`gas_thermal_conductivity_field` from a similarly named material variable.
For an original Talbot-model reproduction, use the translational conductivity
for polyatomic gas, as specified in the original paper's theoretical-background
footnote ([Talbot et al., 1980](https://escholarship.org/content/qt22f5r6cz/qt22f5r6cz.pdf)).
An independently justified simple dilute-gas closure is
`k_tr=(15/4)*mu*(k_B/m_molecule)`; using it is an external reconstruction with
its own validity and uncertainty, not a universally valid mixture rule.
Record whether the source conductivity is translational, total/internal-mode,
species-resolved, or an effective material property, and its frame/domain.

The [COMSOL Thermophoretic Force documentation](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.48.html)
describes a fluid conductivity input; inspect the selected native expression and
material/function binding to determine its actual meaning. If native execution
uses total k, reproducing that expression with total k may be equation-equivalent
to that native variant while differing from original Talbot physics. Keep those
two claims separate. If original physics is the requested question, select or
reconstruct k_tr and register the native model-form difference; never silently
substitute k_tr in a native-reproduction comparison. A conductivity change is
not a mean-free-path convention conversion.

Shared neutral-gas input names must also have a shared physical convention.
Choose one canonical lambda definition compatible with any selected model whose
coefficients are fixed. The Allen–Raabe air revision has fixed drag coefficients;
keep its calibrated air mean-free-path convention. NIST describes the kinetic
relation `mu=phi*rho*c_bar*lambda`, with `phi=0.491` for its hard-sphere relation,
and a calibrated `lambda0` with pressure/temperature scaling
([NIST §3.3, equations 18–20](https://nvlpubs.nist.gov/nistpubs/jres/111/4/V111.N04.A01.pdf)).
Check the actual lambda0, viscosity, gas definition and temperature/pressure
law; phi alone does not reproduce every calibrated air table.

For Talbot the original `lambda_old=2*mu/(rho*c_bar)` with radius Kn can be
expressed using `lambda_common` without changing the force equation if
`alpha=lambda_old/lambda_common` is a known constant over the entire selected
space/time scope. Use the current case's explicit coefficient inputs:

```text
Cm_common = alpha * Cm_old
Ct_common = alpha * Ct_old
Cs_common = Cs_old
```

This preserves `Cm*Kn` and `Ct*Kn`; no conductivity, particle radius, prefactor
or force fit changes. For the particular common relation
`lambda_common=mu/(0.491*rho*c_bar)` with the very same primitives,
`alpha=2*0.491=0.982`. That number is a derivation example, not a preset for
arbitrary COMSOL or calibrated lambda0 data. Record both definitions, ratio
proof, old/new coefficients, and independent point-force checks across Kn and
conductivity ratio. Combine radius/diameter conversions algebraically only
after identifying both definitions.

If the ratio varies with position, temperature, species, pressure or time,
constant scalar Cm/Ct cannot preserve the original equation globally. Register
`UNRESOLVED` when the definitions are missing; otherwise select a supported
common-model slice, a quantified `APPROXIMATE` case, or `UNSUPPORTED` composition.
Do not sample a convenient ratio, average it, or duplicate lambda fields under
different category names to bypass shared-background validation. A new spatially
varying coefficient capability is a core model decision; deriving compatible
primitives and a constant convention conversion is external preparation work.
For other laws, inspect every occurrence of lambda, including applicability
gates; Talbot's algebra is not permission to rescale arbitrary model coefficients.

## 3. Particle, release, and boundary authority

The v3 `tools.comsol_adapter` reads selected static nodal mappings. Domain scope
and finite field support are separate facts: first bind the export's selected
domain/side and mesh identity, then check every required primitive on that scope.
Prefer scoped node/domain IDs. For an older multi-domain coordinate CSV, an
explicit validity anchor is usable only with hash-bound inspected evidence that
it identifies the selected producer domain, followed by mesh coverage and strict
coordinate bijection. It is not a general NaN-to-support rule. Reject surviving
duplicate coordinates or missing values inside that scope; retain outside raw
rows/hashes without filling them. Converter success does not establish field
convergence, axis regularity, `run` preparation or trajectory agreement.

Map independently:

- `mass_kg`: inertia; if derived externally from diameter and density, preserve
  that derivation and source. Solver must not reconstruct it during execution.
- `drag_diameter_m`, `electrostatic_radius_m`, `displaced_volume_m3`, and
  `contact_radius_m`: distinct physical roles. Point contact (`0`) versus finite
  XY disk/RZ sphere contact changes the event surface; it is not a tolerance.
  Use the inspected contact rule; never set contact radius to half the drag
  diameter merely because the source specifies a particle diameter. Verified
  native point contact maps to `0`; finite contact is a declared model change.
- `charge_number`: dimensionless real Z; convert source charge [C] by `Z=q/e`.
  Do not swap particle potential [V], charge [C], current [A], and Z.
- `model_weight`: represented particle count; never multiply acceleration by
  statistical weight or treat it as particle mass.
- Stable particle ID, release time, realized position/facet and velocity:
  export actual per-particle values. Distribution realization and source RNG
  belong to external preparation. Source order/index is not stable identity.

Internal table releases must be strict interior. Surface tables reference the
canonical facet and a strict-interior facet parameter, and carry the realized
velocity. Do not move an on-wall point to make it an internal release.
Surface departure restrictions depend on the chosen integrator; inspect the
current source and run preparation. In Brownian runs zero/tangent surface
velocity is rejected; use the actual physical release distribution, never a
post hoc inward nudge. Release after start time is a local particle interval,
not a reason to change every particle's macro grid.

Resolve the source boundary behavior before choosing a law. Current material
laws are `stick`, `escape`, `hold`, `specular`, `restitution`, `maxwell_thermal`,
and `probabilistic_stick` with an explicit nonstick response. `stick` saves zero
velocity; `hold` preserves hit state, including velocity/charge. Determine
whether the source's frozen trajectory continues charging or reporting an
auxiliary state before calling it equivalent to either. Disappearance needs a
cause and boundary/particle ID; a missing row is not proof of escape.

Assign each material group one law and priority, plus parameters/contact mode
required by the current schema. Keep electrostatic boundary potential, gas
inlet/outlet condition, and particle boundary response separate. Equal voltage
does not justify merging distinct particle groups. `r=0` is the RZ axis and
never a material wall. Pure-translation periodic XY topology has a separate
current selector and seam checks; paired groups do not receive wall laws.
Static geometry can use the current `maxwell_thermal` wall-frame response with
constant two-component velocity tangent to every facet in its group, including
that law's `probabilistic_stick.otherwise` branch. This does not move geometry.
Normal wall motion, moving geometry, arbitrary rotation, 3-D walls and
self-consistent feedback remain explicit gaps; check the current schema before
mapping another wall-frame law.

Inspect internal boundaries separately. A gas/material or mesh-domain interface
is transparent to particles unless the actual selected particle feature says
otherwise. Conversely, a selected COMSOL internal particle Wall is a physical
event even when both neighboring regions are gas. Current material geometry
accepts outer-domain facets; retaining both sides while applying a law on an
internal facet is not represented by relabeling that facet as an outer wall.
Selecting just one accessible side can create a valid outer wall only when that
matches the physical question; otherwise classify the capability gap or register
a justified restricted/approximate case. Gas/electric interface conditions alone
never supply that particle response.

## 4. Geometry and field acceptance gates

Read the current
[case format](../../../../particle_platform_redesign/solver/docs/case_format_v3.md),
[fields.py](../../../../particle_platform_redesign/solver/src/chamber_particles/fields.py),
and [geometry.py](../../../../particle_platform_redesign/solver/src/chamber_particles/geometry.py)
for exact scope. The active supported path has these requirements:

1. **Coordinates and dimensions.** Static 2-D Cartesian XY or axisymmetric RZ
   data paired with XY or RZ-meridional motion. Record geometry length units,
   origin, basis, component order, and velocity meaning; RZ radii are nonnegative.
   RZ is not a flat XY
   interpretation, nor a full 3-D/swirl particle state. A justified reduced
   section of a 3-D source is an approximation with its own scope.
2. **Volume and boundary topology.** Canonical `tri3`/`quad4` volume and `line2`
   outer material boundary. Validate index/order/orientation, finite nondegenerate
   cells, positive Q1 mapping, owner cell, duplicates, non-manifold incidence,
   intersecting edges, and local float64 resolvability. Include all external
   edges exactly once except the RZ axis seam; internal interfaces are not outer
   walls. Preserve holes, disconnected accessible regions, and opening IDs.
3. **Curved/high-order conversion.** Refining a curved CAD wall into line
   segments and sampling higher-order FE fields into P1/Q1 are numerical
   approximations. Record wall displacement/normal error, mesh resolution,
   and interpolation change. Preserve topology; do not make the chord cross a
   narrow gap. An arbitrary high-order element is not a canonical Q1 quad.
4. **Required common layout.** Selected primitives are node-associated and
   share one layout. Fully supported regular layouts must cover every geometry
   node. For P1/Q1 with material geometry, nodes/connectivity must exactly match
   the geometry, which is correspondingly all-triangle or all-quad; a mixed
   source requires an explicit conversion. Required cell support is complete.
   Boundaryless stage-evaluated P1/Q1 motion is unsupported; boundaryless
   stage-evaluated motion requires a fully supported regular box. The design's
   independent mesh owners do not imply arbitrary mesh pairs currently run.
5. **Nodal correspondence and sides.** Join stable topology IDs. If IDs are
   missing, a documented bijection with exactly-one coordinate correspondence
   can be a narrow adapter operation; never nearest-neighbor fill. Domain
   interfaces require an explicit particle-accessibility decision. When only
   one side is accessible, select its one-sided solution and exclude the other
   domain; do not average through the excluded solid. When particles cross a
   transparent interface, shared-node P1/Q1 interpolation has one value per
   node and cannot retain two different limiting values. Canonical regular
   interpolation is also continuous between its shared nodes. Register such a
   traversed jump as `UNSUPPORTED` representation, or make a stated smoothing/
   projection approximation with transition width, off-node side errors and
   sensitivity in the affected observables. Duplicating coincident geometry
   nodes is not a supported interface mechanism. Smaller particle dt does not
   repair the transfer error. Distinguish a discontinuous value from a continuous
   value whose derivative has a jump; the latter can be approximated by continuous
   fields, with derivative recovery accuracy evaluated separately.
6. **Field meaning and spot checks.** Compare raw and canonical values at
   stable nodes plus independent interior locations, domain edges/axis, strong
   gradient regions, and source locations. Check min/max/units/signs, support,
   and a known vector direction. Use appropriate absolute SI and relative
   error budgets away from zero. A nodal copy equality does not assess interior
   high-order-to-linear error. Record nodal and off-node results separately.
7. **Axis regularity.** When the axis is accessible, required RZ vectors must
   have exactly zero radial component at layout axis nodes; selected vorticity
   scalars marked `zero_on_rz_axis` must also vanish. A physical nonzero value
   is not roundoff. If a narrow producer adapter projects bounded numerical
   residue, record its limit, count, and maximum correction. Do not add a
   solver clamp to accept a nonaxisymmetric field.
8. **Time.** Static `values[N,C]`, or finite strictly increasing snapshot
   `time_s[T]` with `values[T,N,C]` on fixed topology. Every required temporal
   field must cover the full run; stages interpolate linearly, with internal
   snapshot knots included in the macro partition. Do not hold/extrapolate
   endpoint values or interpret a parameter sweep as a time series.

Keep a compact `geometry_field_checks` record: source/canonical node/cell/
boundary counts and ID coverage, domain area (and `2*pi*r` volume in RZ when
physically relevant), boundary-group lengths, source containment/clearance,
support coverage, units/basis, snapshot coverage, node spot checks, off-node
errors, and introduced approximations. Geometry visual overlays are a useful
cross-check of that record, not a substitute for topology evidence.

## 5. Derived primitives and reduced-field limitations

For a derivative or closure, retain source solution hash, exact expression,
side, recovery/discretization rule, and units. Check an affine/manufactured
field or independent source evaluations before trusting the transformation.

- Kelvin versus thermal voltage: `V_T=k_B*T/e` only when the source meaning is
  thermal energy; an effective drift temperature is another model assumption.
- `grad(T)` [K/m] and translational heat flux [W/m^2] are different primitives;
  `q_tr=-k_tr*grad(T)` needs a valid translational Fourier closure, conductivity
  authority, and its regime evidence. Total heat flux cannot substitute.
- Signed XY vorticity is `d(u_y)/dx-d(u_x)/dy`; signed RZ azimuthal vorticity is
  `d(u_r)/dz-d(u_z)/dr`. Preserve basis and sign rather than using its norm.
- DEP consumes `grad(<|E|^2>)`, not E or `grad(V)`. Record DC/RF frequency,
  peak/RMS, averaging, gradient recovery, and point-dipole radius certification.
- Mean free path, screening length, effective mass, and species velocity need
  their own closure definitions. A scalar speed cannot recover vector direction,
  and averaged molecular/ion mass does not uniquely reconstruct species sums.

The current CSV adapter consumes an explicit selected-field mapping for a
static single-domain XY or RZ nodal layout. It records native scope, solution,
association, units and basis; it does not infer a domain from arbitrary NaNs.
Read its current README and domain-witness requirements before use. The
adapter's `domain_scope.evidence` is retained caller text, not a dereferenced
or authenticated evidence file. Preparation verifies the actual witness hash
and native selection/export-side meaning using the existing inspection record.
Canonical mesh coverage or equality to an earlier converted case alone does
not authenticate that native domain scope. With unresolved evidence, report
only a declared numerical conversion/analogue, not native-field reproduction.
The
current reduced electrostatic builder is static single-domain RZ, exact P1
layout, Boltzmann-electron/collisionless Bohm-ion closure with specified boundary
potentials and producer parameters. It does not provide arbitrary COMSOL plasma,
negative ions, species transport, RF coupling, or magnetic fields. Its converged
linear/nonlinear solve does not establish that its closure matches the source.

Missing derivative/recovery/producer layouts are external tool issues. A new
physical equation, state, geometry capability, or integrator belongs to its
canonical core owner only after an independent verification case and the
repository's decision workflow. Do not insert model-specific input fallbacks.

## 6. Interpret meaning preflight in the requested scope

The existing `tools/vv/comsol/meaning_preflight.py` validates bound meaning for
two trajectory-comparison questions. It is not the canonical case loader and
does not decide whether a background-field conversion succeeded. Read its
current README/schema when constructing an inventory; mappings in the skill
ledger and the tool's four classifications are related but not identical.

After discovery/mapping, construct a schema-v2 `semantic_inventory.json` only
when the required native source provenance and expected/observed bindings are
available. Missing solution/dataset identity, hashes or readback evidence leaves
a draft with a targeted extraction request; never invent bindings or promote a
producer's status declaration into observed evidence. Plan/conversion work can
continue for the supported scope without claiming a certified comparison.

```console
uv run --locked python tools/vv/comsol/meaning_preflight.py <semantic_inventory.json> --output <fresh-preflight-directory>
```

Interpret the generated JSON and process result together:

| Requested scope | Decision |
|---|---|
| Field/geometry preparation, no trajectory comparison | Both `comparison_conditions` questions have `requested:false` and outcome `NOT_APPLICABLE`. A valid generated report has `requested_questions_ready:false` and the current CLI exits 2 even if every relevant layer is `SUPPORTED`. This does not mean conversion failed. Check `overall_classification`, each relevant `layers` item and its `scope`, and `blockers`; complete required adapters and geometry/field checks before publishing checked inputs. |
| One trajectory question | Require that question's classification `SUPPORTED` before its qualified parity/reproduction PASS/FAIL. Descriptive differences, practical observable agreement and independent candidate verification are separate claims. The overall classification may differ when another layer or unrequested question needs an adapter; inspect the question's own blockers. |
| Two trajectory questions | Each question must be separately supported, bound and evaluated. Identical-canonical-field parity excludes field-production error; native-FE reproduction includes it. |
| Exception, invalid inventory, missing/stale artifacts, or no valid generated JSON | Treat this as missing/invalid evidence, not the benign no-question exit-2 case; retain a draft and diagnose the concrete error. |

The detailed `comparison_conditions.json` owns normalized `layers` and requested
flags; `comparison_summary.json` owns compact `blockers`,
`overall_classification` and `requested_questions_ready`. An excluded
`NOT_APPLICABLE` item is different from a required unsupported item with the same
classification. A relevant `AMBIGUOUS` item blocks its checked mapping; an
`ADAPTER_REQUIRED` item permits doing the adapter work, not declaring it already
done. Rebind and rerun preflight after actual conversion/readback when comparison
certification is required. Background-field/geometry-only preparation completes
with canonical artifact readback, hashes, and geometry/field acceptance; it does
not require a particle case or `check`. Use `check` when an actual particle case
with supplied source requirements is part of the deliverable, and runtime
preparation only for requested execution. No COMSOL particle solution means
trajectory agreement remains unrequested or `NOT_TESTED`, not a failed physical
comparison.
