# Release Packaging Policy

This policy keeps release-candidate packaging predictable without turning the
repository into an artifact store.

## Canonical Documentation

Active productization material lives under:

```text
docs/productization/sim_rev3/
```

Use that tree for release notes, audit records, V&V workflow docs, and reusable
templates. Historical planning material is quarantined under:

```text
docs/productization/sim_rev3/archive/planning_docset_v2/
```

The archive is retained for context only. Do not add new release instructions
or active templates there.

General operator docs stay at top-level `docs/`:

- `docs/architecture.md`
- `docs/canonical_input_bundle.md`
- `docs/comsol_onboarding.md`
- `docs/comsol_parity.md`
- `docs/numerics_contract.md`
- `docs/visualization_workflow.md`

## Templates

The canonical template directory is:

```text
docs/productization/sim_rev3/templates/
```

It contains the active acceptance matrix, task record template, COMSOL minimal
manifest template, compare summary schema, first-step and force contribution
schemas, and cleanup inventory template. Do not create parallel template copies
under `vv/` or archived planning folders.

## Examples

Tracked examples should be small, runnable, and source-like:

- YAML run configs
- small CSV input tables
- small curated NPZ provider bundles used by examples or tests
- README files explaining operator commands and expected compact artifacts

Examples should not contain generated output roots, GIFs, large plots, local
scratch directories, or full COMSOL model files.

## COMSOL Asset Policy

The solver package consumes exported YAML, CSV, and NPZ artifacts. It must not
consume `.mph` files or call the COMSOL API from `particle_tracer_unified`.

Tracked in the repository:

- small COMSOL-free examples;
- small exported CSV/NPZ fixtures that are required by tests or onboarding;
- existing legacy `data/*.mph` and `.mphtxt` files until a separate migration
  decision removes or externalizes them.

External to the repository:

- new `.mph`, `.mphbin`, `.mph.lock`, and large `.mphtxt` files;
- full COMSOL/VIGUS trajectory exports;
- full validation run outputs;
- generated plots, GIFs, decks, reports, and local `_tmp*` or `_out*` roots.

If a large reference asset is needed for validation, store it in external
artifact storage and commit only the manifest, schema, checksum, or operator
instructions needed to locate it.

## Console Scripts

Product entry points should cover normal runs and comparison workflows:

- `particle-tracer-run`
- `particle-tracer-field-compare`
- `particle-tracer-acceleration-compare`
- `particle-tracer-trajectory-compare`
- `particle-tracer-boundary-compare`
- `particle-tracer-first-step-compare`
- `particle-tracer-comsol-full-diagnostics`
- `particle-tracer-build-comsol-case`
- `particle-tracer-compare-reference`
- `particle-tracer-collect-summaries`
- `particle-tracer-validate-artifacts`
- `particle-tracer-export-visualizations`
- `particle-tracer-residual-gap-summary`

Repository tools that are not packaged as console scripts may still be run with
`py -3 tools/<name>.py`. Promote a tool to a console script only when it is
stable enough to be part of the product interface.

## Ignore Rules

`.gitignore` should exclude generated outputs and large external artifacts
while allowing source docs, templates, and examples to be tracked. Use narrow
patterns for known generated roots and broad patterns for COMSOL binary model
artifacts that should not be newly committed.
