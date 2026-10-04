# P18-C external charge evidence

`frozen_rate_parity_v2/` is the current no-COMSOL-rerun replay of the 12
existing saved histories (397,820 active rows). The external evaluator calls
the production aggregate charge model at saved active primitive states and
compares its rate with the exported rate. Its strict core-plus-screening gate
is `FAIL` (`1.8855e-9` maximum current-scale residual versus `1e-10`), while
the same formula using the exported single-charge potential is `PASS`
(`1.5331e-12`). The inferred/exported `epsilon0` convention differs from the
core by `6.8066e-10`, consistently explaining the systematic `phi1` shift;
the report therefore marks model-form mismatch as not indicated.

`frozen_rate_parity_v1/` is the immutable first direct-core run with a stricter
`1e-12` trial gate and without the layered constant-convention diagnosis. It is
retained as historical evidence and superseded by v2 for interpretation.

This evidence is limited to frozen-state formula parity. It is not a golden
dataset, an integrated-charge comparison, a trajectory comparison, or a
continuous-path applicability certificate.
