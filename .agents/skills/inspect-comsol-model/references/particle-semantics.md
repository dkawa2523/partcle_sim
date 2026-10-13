# COMSOL particle semantics that change the comparison

Read the rows relevant to the selected model. These are interpretation prompts
backed by versioned documentation, not defaults or claims of repository model
support. Confirm the exact model's expressions and installed version. The
solver correspondence and applicability belong to
[model-mapping.md](../../prepare-comsol-field-case/references/model-mapping.md).

## Interface and motion formulation

| Native interface/setting | Required interpretation before mapping |
|---|---|
| Mathematical Particle Tracing | Identify the actual motion formulation, dependent variables and custom equations; an interface name alone does not specify a physical particle. |
| Particle Tracing for Fluid Flow | Inspect drag law, rarefaction/slip and wall corrections, Brownian, thermal/lift/gravity terms, turbulence and fluid ownership separately. |
| Charged Particle Tracing | Inspect electric/magnetic force, charge source, collisions, relativity, out-of-plane motion, interactions and field feedback. A plasma field interface is a background producer, not automatically the particle charge law. |
| Newtonian / Newtonian first order | An inertial formulation; compare mass, velocity state and actual force assembly despite different solver representations. |
| Newtonian, ignore inertial terms | A reduced inertia-free balance. Do not replace it with inertial motion and call the equations identical. Validate its time-scale limit separately. |
| Massless | Transport by the prescribed velocity, with different state/equations from an inertial force model. |

COMSOL documents these formulation distinctions and the relaxation-time
criterion for neglecting inertia. [Choosing a Formulation](https://doc.comsol.com/6.3/doc/com.comsol.help.particle/particle_ug_modeling.05.09.html)

For XY/RZ inspect **Include out-of-plane degrees of freedom** and actual DOF
names. Azimuthal motion in RZ can affect radial motion through centrifugal
terms. Projecting a nonzero third velocity component into a meridional plane
changes the problem. Record a deliberate reduction instead of declaring
coordinate equivalence. [Charged Particle Tracing interface](https://doc.comsol.com/6.3/doc/com.comsol.help.particle/particle_ug_acdc.07.02.html)

## Releases and initial velocity

Read the active release selector before its stored properties: an expression
release can retain inactive temperature, cone, momentum, kinetic-energy and
size-distribution values. Their presence is not evidence that they are used.
Bind release feature/selection, position unit, all-combinations versus paired
coordinates, release times, multiplicity and actual stable IDs.

For velocity bind the selected vector, momentum, energy/direction, fluid-following
or distribution rule; component order/basis, direction normalization, absolute
versus fluid-relative velocity, and frame convention. Trace expression dependencies
to their actual solution/member/time. A gas-speed magnitude with a prescribed
direction does not initialize the gas velocity vector. Labels such as a speed
multiplier or an angular range do not specify a speed in SI or an angular PDF.
COMSOL supports several expression/distribution modes and both deterministic and
random sampling. [Release](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.17.html)

For boundary releases inspect the chosen coordinate system and local normal/tangent
orientation. COMSOL's inlet normal may point into the accessible domain, opposite
the geometry normal. Thermal boundary injection and volume Maxwellian sampling
are different laws. [Inlet](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.19.html)

`SubtractMovingFrameVelocity=0` alone does not prove laboratory-frame velocity:
COMSOL initializes relative to the moving frame when the subtraction is cleared.
Inspect active frame dynamics and offsets; unsupported rotation changes both
initial data and governing equations. Capture actual released vectors when available.
For replay, retain realized rows rather than assume that an equal seed or a new
sampler reproduces COMSOL's implementation. Bind position, velocity, release time,
mass, size authorities, charge and weight together, preserving their correlations.
Physical conversion and the current solver's limits are covered by
[initial-conditions.md](../../prepare-comsol-field-case/references/initial-conditions.md).

## Force and charge audit

| Term | Read and resolve |
|---|---|
| Drag | Continuum base law, rarefaction switch and selected slip/free-molecular law; gas velocity/density/temperature/viscosity, mean-free-path definition, molecular species, accommodation, near-wall/turbulence corrections, and size convention. Compare coefficient/equation at matched states rather than feature labels. |
| Brownian | Noise amplitude law and units, gas temperature/viscosity or other background inputs, fluctuation-dissipation partner, random argument mode/feature arguments, time discretization and wall/release handling. Equal seed integers do not share RNG streams. |
| Electric / magnetic | Actual force expression and charge variable; sign, charge units, E/B component basis, static/time/phasor convention and whether all selected terms use the same charge state. |
| Gravity / buoyancy | Acceleration vector versus force, particle mass versus density/volume, gas density and whether buoyancy is included once. |
| Thermophoresis / lift / DEP / ion drag | Actual law/constants/applicability, gradient/recovery and species/transport properties. Similar names and scalar magnitudes do not establish equation equivalence. |
| Custom force / auxiliary ODE | Full expression dependency graph, selection, units, sign, initial/reset conditions and assembly into the particle equation. Check for double counting a built-in and custom force. |
| Particle charge | Fixed charge versus a particle auxiliary state; `dQ/dt` in A versus `dZ/dt` in 1/s (related by elementary charge), individual current signs/species and potential convention. A wall charge accumulation variable is a different state. |
| Particle temperature / mass evolution | Whether auxiliary equations are active; actual initial state, heat capacity, heat sources/sinks, evaporation/growth and force dependence on these states. Disabled temperature computation does not establish `T_particle=T_gas`; density does not identify material heat capacity or surface energy accommodation. |

Gas-molecule accommodation at a particle surface and the tracked particle's
sticking/scattering probability at a chamber wall are different interactions.
Do not transfer a probability between them because both mention reflection.

COMSOL's drag documentation distinguishes continuum and rarefied options and
their inputs. [Drag Force](https://doc.comsol.com/6.3/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.06.html)
Its rarefied theory defines Kn with particle diameter. Compare this with the
repository model's stated radius or diameter convention before translating
thresholds or coefficients. [Rarefied drag theory](https://doc.comsol.com/6.3/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.37.html)
Version 5.5 changed the earlier Epstein/Basset/Phillips Kn convention, so
re-solving an older model under a newer version can change its reference.
[COMSOL 5.5 release notes](https://doc.comsol.com/5.5/doc/com.comsol.help.comsol/COMSOL_ReleaseNotes.pdf)

COMSOL Brownian discretization interacts with the wall accuracy order and
release/collision handling; capture these settings before interpreting a
step-size effect as a change in physical diffusion.
[Brownian Force](https://doc.comsol.com/6.3/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.08.html)

Audit mechanisms outside a one-way fixed-background problem: bidirectional
field coupling, particle interactions, secondary emission, collisions,
variable mass/size, moving geometry and relativistic equations. Preserve them
in the extraction record even when a comparison excludes them. RF cycle
averaging is an explicit model reduction: averaging primitives before a
nonlinear force or charge law need not equal averaging its result.

## Wall and outlet behavior

For standard Particle Tracing for Fluid Flow, COMSOL detects particle-boundary
interactions at the particle's point position; specifying diameter for forces
does not move contact outward by a radius. Preserve that convention when
preparation selects the candidate geometry/contact model. A custom clearance,
offset geometry or interaction law changes the convention and must be inspected
separately. Do not generalize this documented FPT behavior to an uninspected
custom interface. [Particle-wall point detection](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_modeling.05.05.html)

| Native action | State/observation meaning to preserve |
|---|---|
| Freeze | Retains position and velocity at contact; a nonzero stored velocity does not mean the frozen particle continues moving. |
| Stick | Follows the wall after contact; for a stationary wall the retained velocity is zero. |
| Disappear | Later coordinates are absent/NaN; absence alone does not identify which boundary or cause removed it. |
| Bounce | Specular velocity reflection; inspect restitution or added custom rules rather than infer them from the name. |
| Diffuse / isotropic / mixed | Distinct angular laws; inspect postcollision speed, probability and alternative branch. Do not confuse diffuse angular scattering with thermal reemission. |
| Thermal reemission / general reflection | Inspect wall temperature or specified velocity and its normal/tangent basis. |
| Inlet / outlet | Release law and return-to-inlet law are separate. Outlet can freeze or disappear. |

These state rules and reflection distinctions are described in
[Boundary conditions](https://doc.comsol.com/6.3/doc/com.comsol.help.particle/particle_ug_math.06.38.html).
Inspect interior versus exterior selection, active overrides, conditional
Probability/Expression and Otherwise branches, secondary emission and moving
wall settings. The default Wall selects exterior boundaries; an explicit Wall
can select internal boundaries. [Wall settings](https://doc.comsol.com/6.3/doc/com.comsol.help.particle/particle_ug_math.06.04.html)

## Status, events and identity

When saved, `<scope>.rti`, `.st`, `.fs` describe release time, stop time and
final status. During postprocessing `.fs` refers to the final saved time,
not each requested time. Timewise `particlestatus` needs stored status data.
Native codes are 0 unreleased, 1 in-domain, 2 frozen, 3 stuck, 4 disappeared;
these are not the repository's lifecycle codes. Read the storage settings and
map meanings explicitly. Extra wall-time storage assists plots; it does not
guarantee a complete rebound event ledger. The wall-interaction-limit safeguard
can also remove particles, so disappearance is not automatic outlet escape.
DOF reuse for secondary particles can invalidate static metadata and stable
row identities. [Mathematical Particle Tracing interface](https://doc.comsol.com/6.3/doc/com.comsol.help.particle/particle_ug_math.06.02.html)

Check available identity variables such as `<scope>.pidx`, release feature and
release time. Include physics scope and sweep member in the identity map.
Primary-release statistics can omit secondary particles. With DOF reuse,
recover generation/release identity or mark particle matching ambiguous.
[Special Variables](https://doc.comsol.com/6.3/doc/com.comsol.help.particle/particle_ug_modeling.05.11.html)

Saved time lists are observation times. They do not generally identify accepted
internal integrator steps or exact reflection times. Treat finite-difference
velocity, inferred event brackets and incomplete particle states as limited
observations with error/coverage, not native exact states. Candidate time
selection belongs to the independent
[numerical plan](../../prepare-comsol-field-case/references/numerical-plan.md).

When comparing native accumulated results, bind the selected particle cohort,
source term/unit, Count versus Density, Elements versus Elements and time,
and Source interpolation settings. Instantaneous occupancy, time accumulation
and physical concentration are different quantities. A particle trajectory
plot can have dataset/filter/cohort restrictions; its visible lines need not
be the complete population. [Domain accumulator](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.15.html)
