# COMSOL Faithful 2D Template

This directory shows the expected shape of a `comsol_faithful` case. Replace the
`generated/*.npz` placeholders in `run_config.yaml` with field and geometry
exports produced by the external COMSOL export tools.

The manifest is intentionally the single comparison metadata source. Release
particles, boundary entity mapping, wall laws, coordinate scale, field physical
quantities, and force inventory all live in `comsol_case_manifest.yaml`.
