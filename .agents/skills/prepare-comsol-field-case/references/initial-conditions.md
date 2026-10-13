# Initial conditions: physical meaning and realized input

Use this reference when preparing or diagnosing particle releases. Native active
selectors and evidence extraction belong to
[particle-semantics.md](../../inspect-comsol-model/references/particle-semantics.md).
Current support belongs to the repository schema/source code, not this guide.
The objective is a physically justified tracking problem; reproducing an external
implementation is a separately scoped verification exercise.

## Bind the joint source, not just its positions

One source record should link native or requested meaning to realized rows:
stable particle ID, release feature/time/member, position and unit, velocity and
basis/frame, mass, drag diameter, electrostatic/contact radius, volume, charge,
weight and sampling method. Preserve correlations among these quantities. Record
active selectors, expression dependencies and the particle count actually realized.
Do not reconstruct IDs from undocumented grid ordering or Cartesian products.

For source-expression replay evaluate at each release's position and physical time,
using the bound field/member. Complex native expressions should be evaluated by
their producer and exported, not translated through guessed variable aliases or
a general expression evaluator. An exact saved realization can be transferred even
when reproducing the native sampler algorithm is unnecessary or unavailable.

## Convert velocities with explicit conventions

| Source meaning | Canonical realization and important distinction |
|---|---|
| Components | Convert SI and transform the declared basis to the case basis; preserve signs/component order. A scalar speed is insufficient. |
| Speed and direction | `v=s*l/|l|`; reject a zero direction. Distinguish spatially varying deterministic directions from an angular distribution. |
| Momentum | For nonrelativistic inertial motion, `v=p/mass_kg`; use the actual inertial mass. Relativistic momentum is a different equation, not this conversion. |
| Kinetic energy and direction | For the same formulation, `s=sqrt(2*E/mass_kg)` with energy in joules; normalize direction. Bind the active energy mode before reading stored energy values. |
| Fluid following | `v=u_g(x,t)`; adding a supplied relative vector instead means `v=u_g+w`. A multiple of `|u_g|` in a prescribed direction is neither of these. |
| Boundary normal/tangent | Resolve the accessible side and oriented orthonormal basis first; convert the full vector and check its dot product with the domain outward normal. Never infer orientation from a positive native normal component. |
| Moving frame | Translate/rotate basis and velocity using the actual frame kinematics. A one-time vector conversion is insufficient when the motion equation includes rotating-frame terms unsupported by the solver. |
| XY/RZ with a third component | Preserve its evidence. Current two-component tracking does not support its independent dynamics; RZ azimuthal velocity can change radial motion. Require an explicit reduction or unsupported verdict, never silent projection. |

The formulas in this table assume nonrelativistic inertial particles. Inertia-free
or prescribed-velocity formulations require their own model correspondence;
an exported initial vector does not turn them into the same dynamical problem.

## Distribution and measure traps

For independent physical source generation, define a probability law and its
measure, then materialize it outside the core. Examples below are mathematical
laws, not promises of current source-builder capability or COMSOL sampler identity.

- A volume thermal velocity in `d` retained Cartesian components has
  `v_i-u_i ~ Normal(0,k_B*T/m)` independently; `E[|v-u|^2]=d*k_B*T/m`.
  The speed PDF depends on `d`. Uniform speed plus random direction is not a
  Maxwellian, and `m` is the particle mass, not the gas-molecule mass.
- Isotropic 3-D directions have uniform azimuth and uniform `cos(theta)`.
  A solid-angle-uniform cone uses `cos(theta)=1-U*(1-cos(alpha))`.
  A 2-D uniform arc instead has uniform signed angle. Uniform polar angle in
  3-D biases the distribution. A deterministic quadrature may approximate the
  same law without matching a native pseudorandom realization.
- A constant-speed Lambertian hemisphere has angular density proportional to
  `cos(theta)` per solid angle: `cos(theta)=sqrt(U)` and uniform azimuth.
  A physical equilibrium surface flux is additionally speed weighted.
  With inward normal and zero wall-relative bulk flow, the flux law has
  normal speed `sqrt(-2*(k_B*T/m)*log(U))` and Gaussian tangential components.
  Derive this from `(v dot n_in)_+*f_Maxwell(v)`. Drifted gas, a truncated
  reservoir distribution, prescribed emission and a producer's thermal
  reemission law must each be identified; do not equate them automatically.
  Check the installed producer's evaluated speed law as well as angular law.
  [COMSOL boundary theory](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.38.html)
- Uniform RZ meridional sampling uses `dr dz` or edge `ds`; a physical revolved
  population uses volume `2*pi*r*dr dz` or surface `2*pi*r*ds`. For uniform
  volume in an annulus, sample `r=sqrt(r_min^2+U*(r_max^2-r_min^2))`.
  Declare diagnostic meridional populations separately from physical rates.
- Realized release times are independent of integration/output times. A fixed
  list, a deterministic quadrature over time, and Poisson emissions with rate
  `lambda(t)` are different populations. Statistical physical counts require
  specified rate/exposure and model weights; do not infer these from row count.

Use explicit sampler revision and identity/draw-based external RNG if generating
new rows. Write once, hash and reuse; runtime worker count, tile width, output
schedule and resume must not resample a source. Source RNG is separate from
Brownian and wall streams. No seed equality establishes shared random paths.

## Check the appropriate claim

For realized replay compare every matched ID's release time, position, full
retained velocity vector, charge, mass and independent size/contact/weight
authorities in SI before integration. Report unmatched IDs and unsupported DOF.
An equality test of speed alone misses direction/frame errors. Use appropriate
roundoff/export budgets without fitting the initial vector to later trajectories.

For independent statistical sources compare the joint distribution, conditional
means/covariances, angular/flux moments, release rate and weighted population.
Unrelated random realizations are evaluated statistically, not ID-wise as
paired trajectories. Isolate source uncertainty from dynamics/noise uncertainty.

Current `sources.py` schedules canonical realized rows; it does not own native
expressions or distribution sampling. Use the canonical writer in an external
producer. Surface coordinates bind to facets and contact geometry; velocity
remains in the canonical case basis. Respect contact-radius offsets and the
existing zero-time departure law rather than adding a positional nudge.

## Use the implemented numeric producer when it fits

Read `solver/tools/source_builder/README.md` and use the existing external CLI:

```console
uv run --locked python -m tools.source_builder TEMPLATE.yaml INITIAL.csv SOURCE.json NEW_DIRECTORY
```

It supports numeric vector, speed/direction, nonrelativistic momentum/energy,
explicit SI conversion and static/per-row orthonormal two-component velocity
bases for internal XY/RZ tables. Position already uses the template's coordinate
basis/origin. Missing properties can be inherited only by exact existing particle
ID, after verifying the same bound cohort and every inherited authority; the
same numeric ID in another MPH is no such evidence. For another population,
supply all independent property columns rather than inherit chamber defaults.
The current CLI matches IDs but does not authenticate that provenance. Select actual
release rows explicitly before conversion. The producer keeps numeric correlations,
snapshots its CSV/config bytes for parsing/hashing, and uses the canonical writer
and public `load_case`; it neither solves nor verifies the physical source law.

This producer does not evaluate native expressions, sample thermal/angular/time
distributions, generate surface releases or handle nonzero third DOF/dynamic frames.
`frame: inertial` declares the candidate frame; it is not a way to resolve missing
native evidence. An explicitly scoped same-component numeric replay can retain
unresolved native-frame metadata, but cannot claim physical frame equivalence.
Check all realized arrays and inherited authorities before running; hashes alone
do not establish the desired physics.
