# External gas-collision qualification

This tool checks a prescribed three-dimensional Maxwellian bath and spherical
specular/diffuse scattering independently of the production finite-speed drag
coefficient. It does not advance particle trajectories, locate fields, process
walls, or provide a production noise model.

The CSV requires `label`, `particle_mass_kg`, `molecular_mass_kg`,
`gas_density_kg_m3`, `gas_temperature_K`, `diameter_m`, `diffuse_fraction`,
`surface_temperature_K`, `speed_ratio`, and `gas_mean_free_path_m`.
`speed_ratio` is S=|v-u|/sqrt(2kBT/m). All material and surface quantities are
explicit. The current production comparison assumes equal surface/gas
temperature; a different supplied surface temperature describes a different
reference problem and can intentionally produce a mean-law discrepancy.

```console
uv run --locked python -m tools.vv.gas_collision.qualify --regimes regimes.csv --output qualification --jump-labels low,high --jump-count 400000 --durations-s 2.5e-7,1e-6 --interval-count 50000
```

Choose the durations for the intended physical question. They are frozen
collision-observation intervals and do not set a solver timestep.

The reference uses Gauss-Legendre quadrature for the incident hemisphere,
outgoing flux-Maxwell normal moments, and the shifted spherical Maxwellian
integral. It retains the actual impulse law; it does not call production G(S).
The incident radial domain omits only molecular speeds above 12 sqrt(kBT/m),
whose Maxwell probability is bounded by
erfc(12/sqrt(2))+sqrt(2/pi)*12*exp(-72). The outgoing normal cutoff is 10
sqrt(kBTw/m), with probability exp(-50). Probability-tail bounds alone do not
bound every weighted high-order moment; the reported quadrature refinement and
analytic low-speed checks also matter.

The independent collision sampler draws the flux-weighted incident law
|c-v|f(c). Its rejection proposal is proportional to (|v|+|c|)f(c), so the
triangle inequality gives a valid acceptance bound for the unbounded Maxwell
distribution. The surface normal has cosine-flux weighting. Diffuse outgoing
normal speed is Rayleigh and the tangential components are Gaussian.

Q is the raw jump second moment times the collision rate, in m²/s³. Fokker-Planck
diffusion is Q/2. A compound Poisson increment has third/fourth cumulants even
when its first two moments match a Gaussian increment. The frozen sampling and
characteristic-generator probes quantify that distinction at specified scales;
they do not certify a nonlinear SDE, a wall-arrival tail, or a universal cutoff.
The paired generator-residual standard error retains the correlation between
the exact and truncated expressions evaluated on each same jump sample.

The specular recoil option uses mM/(m+M) only for the specular branch. Diffuse
scattering still assumes a heavy sphere with prescribed surface temperature.
It is not a finite-recoil energy-conserving diffuse model. A one-pseudogas
input is not a species-resolved mixture. A three-component collision moment is
not a qualification of a meridional two-DOF trajectory engine.

The source derivation and model limits are owned by
[technical research §12](../../../../technical_research.md) and the associated
physics design note. The distinction between collision moments and diffusion
trajectory validity is also treated in
[Trigger et al., §III](https://arxiv.org/pdf/cond-mat/0401327).

`seconds` for each conditional-jump record measures its sampling/moment stage;
the report's total `seconds` includes input parsing and all numerical probes,
and excludes the final JSON/CSV write.
Peak traced allocation includes Python/NumPy allocations, and is not a
whole-process RSS measurement. No new JIT or dependency is introduced.
