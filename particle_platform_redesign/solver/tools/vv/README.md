# External verification tools

These tools consume canonical inputs or public solver results. They are
outside the solver core and do not establish a second execution engine.

[COMSOL tools](comsol/README.md) document meaning-matched comparisons,
historical evidence audits, and current candidate replay. Current replay
requires freshly registered format-3 input/template hashes and an explicit
executor identity. Historical hashes are preserved; archived recipes are
not upgraded by changing a format number.

The portable, license-free checks for the current generic preflight and
boundary utilities use synthetic data created in temporary directories:

```console
uv run --locked python -m pytest tools/vv/comsol/tests/test_meaning_preflight.py tools/vv/comsol/tests/test_boundary_response_mapping.py tools/vv/comsol/tests/test_curved_wall_line2_convergence.py tools/vv/comsol/tests/test_prepare_f02_fixed_electric_case.py -q
```

Run the complete external checks locally when the historical reference assets
named by the case configurations are present:

```console
uv run --locked python -m pytest tools -q
```

That complete command includes historical hash audits and replay fixtures
which still read locked metadata, templates, or reference tables from
`model_dataset/` and local `evidence/`. Those assets are not all tracked in
Git, so this is an asset-dependent local gate, not the clean-checkout CI
selection. Current executable fixtures write format-3 canonical data; some
obtain their model parameters or registration metadata from those historical
assets. Public workflow checks use `load_case -> simulate -> open_result`.
Passing either command does not mean that a licensed COMSOL solve or a
native-model comparison ran.
