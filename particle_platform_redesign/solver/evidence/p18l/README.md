# P18-L evidence

`performance_v1.json` records a machine-local, non-gating warm-stage
observation for 100,000 fixed-charge rows. Both variants reuse one allocated
physics workspace. The enabled variant samples two-component gas velocity plus
gas density, mean free path, and signed azimuthal vorticity, then evaluates the
R-Z rarefied-vorticity lift sensitivity in the same compiled stage pass.

Across nine alternating warm repeats, the median was 23.354 ms with lift
disabled and 25.514 ms with lift enabled (1.0925x elapsed time, about 9.25%
stage overhead). Prepared bound storage increased from 1,600,000 to 2,500,016
bytes: one float64 coupling bound and one boolean static-applicability value per
particle, plus the shared two-component gas-velocity bound. At one million
particles this increment is about 9.0 MB. It is prepared runtime memory, not a
new resident particle state or result-schema column.

This is a local implementation observation, not an end-to-end release
benchmark or a claim about other machines. The accepted P14-U evidence remains
the authority for release-scale throughput and memory. Physics correctness is
established separately by the independent three-dimensional cross-product
oracle, scaling and applicability checks, velocity-dependent enclosure tests,
and RK4/exponential-midpoint convergence tests.

The saved COMSOL packages do not contain enough trajectory-local vorticity and
solver-step provenance to certify pointwise lift parity. That external V&V item
was `NOT_TESTED` at P18-L closeout and is not a P18-L core acceptance criterion.
The later M3-C1 common-P1 composite slice passes, but isolated/native-field lift
parity and physical validity remain untested.
