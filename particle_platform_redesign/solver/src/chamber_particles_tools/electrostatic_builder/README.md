# Reduced electrostatic field builder

This installed field producer solves a **static, gas-only RZ/P1 reduced model** from canonical thermal-flow data
and explicit bulk-plasma/boundary assumptions. It is independent of the particle engine, COMSOL and model_dataset.
It does not implement the full-material/floating-conductor V/Q solver described in the separate historical Vigus report.

```console
uv run --locked python -m chamber_particles init electric-example --example reduced-electrostatic
uv run --locked python -m chamber_particles inputs electrostatic check electric-example/builder.yaml
uv run --locked python -m chamber_particles inputs electrostatic build electric-example/builder.yaml electric
```

Python uses `chamber_particles_tools.electrostatic_builder.workflow.check_from_configuration(configuration_path)`
and `build_from_configuration(configuration_path, output_directory)`.
`check` uses the same configuration/input/topology/anchor preparation without solving or publishing.
`build` publishes `bundle.h5` and `report.json` together in a fresh directory; it does not select particle physics or write a second case dialect.
Use the output as a recipe's `input_bundle`, choose models explicitly, and publish through `inputs.create_case`/CLI `create`.

## Supported problem and exclusions

The revision is `reduced_electrostatic_builder_v3`; configuration format is 2.
It requires axisymmetric RZ, one gas domain, a fully supported P1 layout exactly matching triangle-only geometry,
canonical gas velocity/temperature and a potential on every non-axis perimeter facet. Each connected component
must have a potential anchor. Axis facets are symmetry, not material potential or particle walls.

The model has vacuum permittivity, one bulk electron/positive-ion density, Boltzmann electrons and collisionless
Bohm ions, with explicit density/speed floors and C2 sheath smoothing. Boundary groups use fixed potential or a
radial exponential profile with declared priority. No unknown floating voltage or conductor charge is solved.

Mixed geometry, dielectric materials/interfaces, floating conductors, anisotropic epsilon, free sheet charge,
RF/circuit, ion continuity/species transport, negative ions, dynamic wall charging, moving mesh, 3D and two-way
coupling are outside this revision. Unsupported cases are rejected rather than replaced by another closure.

## Equations and numerical meaning

```text
-div(epsilon0 grad V) = e (ni-ne),  E = -grad V
uB = sqrt(e Te_V / mi)
psi = (V - V_bulk) * negative-sheath C2 gate
ne = max(density_floor, n0 exp(psi/Te_V))
ui = sqrt(max(speed_floor^2, uB^2 - 2 e psi/mi))
ni = max(density_floor, n0 uB/ui)
```

`Te_V` is electron thermal voltage [V]; gas/ion temperatures are [K]. The gate uses the explicit smoothing
width and the supported quintic polynomial. Its derivative and active floor branches enter the analytic Jacobian.
Ion vector output follows the explicitly regularized drift-diffusion flux direction:

```text
Di = mobility * kB*Tg/e
Gamma = ni*ug + mobility*ni*E - Di*grad(ni)
u_ion_vector = ui * Gamma/sqrt(|Gamma|^2 + (n0*regularization_speed)^2)
```

The separate scalar positive_ion_speed is the closure's ui; vector magnitude can be smaller because of direction
regularization. This is an explicit reduced-model assumption, not a measured native ion distribution.

P1 Galerkin FEM uses `2*pi*r` volume measure, three-point source/Jacobian quadrature, boundary continuation,
Newton with line search and NumPy matrix-free restarted GMRES. There is no SciPy requirement, alternative solver
fallback or automatic tolerance relaxation. Gas-side nodal gradients use axisymmetric mass-lumped recovery and
exact radial-field zero on the axis. A small algebraic residual is not field mesh convergence or COMSOL parity.

## Complete example configuration

Paths are relative to the configuration. The installed reduced-electrostatic example supplies an editable
`thermal.h5` and `builder.yaml`. A format-2 configuration has the following keys:

```yaml
format_version: 2
input:
  data_path: thermal.h5
  layout: background
  gas_velocity_field: gas_velocity
  gas_temperature_field: gas_temperature
model:
  revision: boltzmann_bohm_sheath_c2_v1
  bulk_number_density_m3: 1.0e+12
  electron_temperature_V: 3.0
  positive_ion_mass_kg: 6.6335209e-26
  bulk_potential_V: 0.0
  ion_mobility_m2_V_s: 1.0
  ion_flux_regularization_speed_m_s: 1.0
  sheath_smoothing_V: 0.05
  density_floor_m3: 1.0
  ion_speed_floor_m_s: 0.001
boundaries:
  - {group: bottom, priority: 1, potential: {kind: constant, value_V: 0.0}}
  - {group: right, priority: 1, potential: {kind: constant, value_V: 0.0}}
  - {group: top, priority: 3, potential: {kind: constant, value_V: 1.0}}
  - {group: left, priority: 1, potential: {kind: constant, value_V: 0.0}}
solver:
  continuation_ramps: [0.0, 1.0]
  max_newton_iterations: 20
  relative_residual_tolerance: 1.0e-9
  absolute_residual_tolerance_C: 1.0e-22
  max_linear_iterations: 100
  linear_krylov_dimension: 20
  linear_relative_tolerance: 1.0e-11
  minimum_line_search_factor: 0.0009765625
```

The numbers and groups are educational example assumptions. A radial exponential potential instead declares
`kind: radial_exponential`, `inner_value_V`, `outer_value_V`, `start_radius_m` and `transition_length_m`.
Conflicting equal-priority boundary values, incomplete perimeter, invalid anchors/material representation and
nonfinite or unsupported input fail before publication.

## Generated fields and dependent particle coefficients

The output preserves upstream fields/sources and adds electric_potential, electric_field, electron_number_density,
positive_ion_number_density, space_charge_density, electron_temperature [K], positive_ion_temperature [K],
positive_ion_velocity, positive_ion_speed and positive_ion_mass. Output-name collisions are rejected.
The report binds config/input/output identity, assumptions, field ranges, convergence and declared solver storage.

If a particle case uses prescribed charge/ion/DEP coefficients derived from an electric/plasma field, regenerate
those dependent coefficients from the new solution before calling it a consistent case. This generic builder
does not implement Vigus-specific coefficient wiring or independently recompute neutral gas flow/temperature.

```console
uv run --locked pytest tests/verification/test_electrostatic_builder.py -q
```
