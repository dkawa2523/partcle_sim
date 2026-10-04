# P18-D evidence

`performance_v1.json` records a machine-local, non-gating warm-stage
observation for 100,000 fixed-charge rows.  Both variants reuse one allocated
physics workspace; the enabled variant samples a non-constant two-component
`gradient_mean_e_squared` field and evaluates
`quasistatic_spherical_gradient_e2_v1` in the same compiled stage pass.

Across nine alternating warm repeats, the median was 22.94 ms with DEP
disabled and 23.83 ms with DEP enabled (1.039x elapsed time, about 3.9%
overhead).  Both prepared runtimes own 1,600,000 bytes of bound arrays, so the
model adds no resident or prepared per-particle array beyond the existing
external-acceleration bound.  The sampled field itself is canonical input and
is not counted as runtime-bound memory.

This is a local implementation observation, not an end-to-end release
benchmark or a claim about other machines.  The accepted P14-U evidence remains
the authority for release-scale throughput and memory.  Physics correctness is
established separately by the analytic formula, bound, applicability-cap, and
RK4/midpoint convergence tests.
