# P18-I evidence

`frozen_force_parity_v1/` is a no-COMSOL-rerun replay of the 12 saved
`model_dataset` histories (397,820 active rows). The external tool first
reconstructs each producer's saved equation and separately evaluates the
production-canonical revision at the same frozen states.

- Native saved-formula replay: **PASS**, global relative L2 residual
  `4.3466e-16`, maximum force-scale-normalized residual `1.7901e-15`.
- Relative-flow production formula: strict `1e-10` comparison **FAIL**,
  global relative L2 residual `1.1601e-9`. The report identifies the saved
  COMSOL `epsilon0=8.8541878188267e-12 F/m` convention versus the production
  SI value `8.8541878128e-12 F/m`; the solver constant and threshold were not
  changed to fit the dataset.
- Image production formula: `DOCUMENTED_MODEL_DEFINITION_DIFFERENCE`, global
  relative L2 residual `0.034648`. Saved Case P uses
  `sqrt(norm(ui)^2+1)` and saved Case A uses a reconstructed `AS_ui_mag`;
  production intentionally uses one producer-independent `norm(ui)` rule.

These are frozen-force formula checks, not integrated trajectory accuracy,
continuous-path applicability, boundary parity, or physical model
certification. The saved dataset is evidence, not a core dependency or golden
truth. `report.json` SHA-256 is
`569264d133b98c215cc8c4ae4044a35565601cd53ce9151ade3ae156d93bf2b6` and
`package_metrics.csv` SHA-256 is
`e9f4bcf01e6faebdc6d8057cbb96be3f5c92a15843d11f1bf8e47ac3e9c5dbd9`.

`performance_v1.json` is a machine-local, non-gating warm-stage observation
for 100,000 rows with P18-C dynamic charge enabled. It reuses one allocated
workspace and reports nine-repeat medians. Relative-flow adds only the
two-component ion-velocity path-bound array; image adds no prepared resident
bound beyond charge-only. This observation does not replace the accepted P14-U
end-to-end release evidence and is not a causal before/after source-code
benchmark.
