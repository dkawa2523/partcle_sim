# P18-R neutral-transport closure v1

Status: `PASS` for the offline audit. This is not a physical mixture-truth or trajectory certificate.

The tool verified the exact 12-package identity, locked hashes, neutral-force settings, export
provenance, and source MPH identity without running COMSOL. It replayed native COMSOL linear
Epstein on 397820 active saved rows; the maximum vector relative residual was
`1.05859e-15`.

Using the saved effective mixture molar mass only as a numerical sensitivity, it characterized
P15-E speed ratio, Kn, and finite-versus-linear coefficient differences. Existing P15-E is
`NOT_APPLICABLE` as physical reference closure because CF4/O2 species/accommodation semantics
and particle surface temperature are unknown.

The Waldmann heat-flux form is algebraically identical to the gradient form after substituting
`q=-k grad(T)`, `m=M/N_A`, and `R=N_A k_B`. Existing P16 saved-row low-speed coverage spans
`0.542834` to `1` across cases, but its single-species
physical applicability is `NOT_APPLICABLE`. The trajectory rows export neither translational
heat flux nor the PPR temperature gradient, so numerical thermophoretic-force replay is
`NOT_TESTED`.

On finite inside-domain background-grid rows, the descriptive `lambda*|grad(T)|/T` range was
`0.000208689` to `0.0860031`. This does not replace
trajectory-local PPR provenance or continuous-path certification.

Decision: `epstein_linear_effective_gas_sensitivity_v1` and
`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1` are required for
same-form reference comparison, with an explicit finite `maximum_speed_ratio` in `(0, 1]`.
They are not species-resolved mixture truth. Continuous-path
applicability remains runtime-certification work and is `NOT_TESTED` here.

`case_matrix.csv` contains all per-package metrics and independent status columns.
